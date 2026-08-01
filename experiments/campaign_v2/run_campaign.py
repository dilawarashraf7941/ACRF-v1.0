"""ACRF v1.0 Final Experimental Campaign — execution script.

Runs the complete campaign specified for this session: Heuristic Policy
(baseline), Cold-Start LinUCB (alpha sweep), Sequential Learning LinUCB
(alpha sweep, via the Sequential Replay Learning Mode added in prior
work), Random Critic / Reduced Context / Quality-only Reward (ablations
vs. Full ACRF).

This script is execution-only: it calls nothing but already-existing,
unmodified ACRF classes — `ExperienceRecord`, `InMemoryExperienceRepository`,
`ExperimentConfig`, `ExperimentRunner`, `AblationConfig`, `AblationRunner`,
`ReplayEngine` (both `.replay()` and `.replay_with_learning()`),
`OfflineEvaluator`, `Benchmark`, `BootstrapExperienceRepository`,
`RandomCriticPolicy`, `ReducedContextPolicy`, `QualityOnlyRewardStrategy`,
`app.evaluation.statistics.Analyzer`, `app.evaluation.learning_analysis.LearningAnalyzer`,
and their exporters/report generators. No new framework code, no new
package under `app/`, no modification to any existing module.

Reproducibility: the synthetic experience log is generated with
DATA_SEED (same seed value as campaign_v1, for continuity); every
bootstrap-based experiment/ablation uses CAMPAIGN_SEED. Both are fixed
constants below.

Post-leakage-fix revalidation note: `generate_repository` was minimally
updated (this file only) to populate `state_features["task_type"]`,
`state_features["max_iterations"]`, and
`state_features["planner_output"]["decomposition"]` -- the three
pre-decision signals `app.evaluation.offline.replay.build_offline_context_vector`
now reads after the target-leakage fix. Previously `state_features` only
carried `error_feature_count`/`worker_output_count`, which the corrected
context builder does not read at all, so every record produced an
identical, all-zero context vector. `task_type` is assigned per
archetype and is causally consistent with `router_node`'s real
`task_type == "code" -> CodeCritic` rule (Methodology Section 3.1-F):
the `CodeCritic` archetype is the only one labeled `"code"`. Decomposition
length is drawn from an archetype-specific mean via its own independent
RNG draw -- never derived from `quality`, `latency`, `iterations`, or
`status` -- so no outcome field leaks into the new pre-decision signals.
`max_iterations` is a fixed constant across every record, matching how
the framework treats it as a run-level configuration budget rather than
a per-record outcome. Because this adds new RNG draws to the generation
loop, every downstream value (not only the three new fields) differs
from the previously published dataset realization under the same
DATA_SEED; this is expected, not a reproducibility break -- see the
change report accompanying this revalidation.

How to run:
    cd <repo root>
    python experiments/campaign_v2/run_campaign.py

Outputs are written to `experiments/campaign_v2/results/` (tables, raw
JSON/CSV/Markdown exports) and `experiments/campaign_v2/figures/` (PNGs).
Re-running reproduces every number and figure exactly.
"""

import json
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.evaluation.ablation import (
    DEFAULT_ALPHA_SWEEP,
    AblationConfig,
    AblationReportGenerator,
    AblationRunner,
    QualityOnlyRewardStrategy,
    RandomCriticPolicy,
    ReducedContextPolicy,
)
from app.evaluation.experiments import ExperimentConfig, ExperimentResult, ExperimentRunner
from app.evaluation.experiments import Exporter as ExperimentExporter
from app.evaluation.experiments.runner import BootstrapExperienceRepository
from app.evaluation.learning_analysis import LearningAnalyzer, LearningReportGenerator
from app.evaluation.offline import ReplayEngine, ReplayStep
from app.experience import ExperienceRecord, InMemoryExperienceRepository
from app.policy import HeuristicPolicy
from app.policy.linucb import LinUCBPolicy
from app.reward import RewardCalculator

BASE_DIR = Path(__file__).parent
FIGURES_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

DATA_SEED = 12345
CAMPAIGN_SEED = 2024
NUM_RUNS = 30
NUM_RECORDS = 300
ALPHAS = DEFAULT_ALPHA_SWEEP  # the framework's own configured alpha sweep: (0.25, 0.5, 1.0, 2.0)
CANONICAL_ALPHA = 1.0
CANDIDATE_ACTIONS = ["LogicCritic", "CodeCritic", "FactCritic", "MetaCritic"]

DEFAULT_MAX_ITERATIONS = 5
"""Fixed run-level iteration budget applied to every synthetic record.

A constant, not a per-record draw: `max_iterations` is a configuration
value in the live framework (Methodology Section 3.1-H), not an outcome,
so it is not varied by archetype or by any other record property here.
"""

# --- 1. Synthetic experience log (seed shared with campaign_v1; generator no
#        longer identical -- see the module docstring's revalidation note) ---

_ARCHETYPES: list[tuple[float, list[str] | None, float, float, float, float, float, str, float]] = [
    # (weight, selected_critics, quality_mean, quality_sd, latency_mean, latency_sd,
    #  iter_mean, task_type, plan_steps_mean)
    (0.40, ["CodeCritic"], 0.78, 0.08, 1.1, 0.25, 0.8, "code", 1.5),
    (0.20, ["LogicCritic"], 0.68, 0.09, 1.4, 0.30, 1.3, "reasoning", 2.5),
    (0.15, ["FactCritic"], 0.60, 0.10, 1.6, 0.30, 1.6, "research", 2.0),
    (0.10, ["MetaCritic"], 0.58, 0.12, 1.8, 0.35, 2.0, "escalation", 3.5),
    (0.15, None, 0.50, 0.13, 2.2, 0.40, 2.8, "multi_domain", 4.0),  # multi-critic episodes
]
"""Each archetype now additionally fixes a `task_type` label and a mean
decomposition length (`plan_steps_mean`), the two archetype-level,
pre-decision properties the corrected offline context builder can read
(`is_code_task`/`has_task_type` from `task_type`, `plan_complexity` from
decomposition length). `task_type == "code"` is assigned only to the
`CodeCritic` archetype, mirroring `router_node`'s actual live routing
rule (Methodology Section 3.1-F) -- this is a legitimate pre-decision
signal, not leakage, because in the real system task type is decided
before routing, not after. The other four labels are distinct strings
for dataset realism; the current 4-feature offline context only
distinguishes `"code"` from everything else, so they are presently
equivalent to each other via `is_code_task`, but are kept distinct so a
future, richer offline context is not artificially prevented from using
them."""


def _clip(value: float, low: float, high: float | None = None) -> float:
    value = max(low, value)
    return min(value, high) if high is not None else value


def generate_repository(seed: int, n: int) -> InMemoryExperienceRepository:
    rng = random.Random(seed)
    repository = InMemoryExperienceRepository()
    weights = [archetype[0] for archetype in _ARCHETYPES]
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)

    for i in range(n):
        _, critics, q_mean, q_sd, lat_mean, lat_sd, iter_mean, task_type, plan_steps_mean = (
            rng.choices(_ARCHETYPES, weights=weights, k=1)[0]
        )
        if critics is None:
            critics = rng.choice([["FactCritic", "MetaCritic"], ["LogicCritic", "MetaCritic"]])

        quality = _clip(rng.gauss(q_mean, q_sd), 0.0, 1.0)
        latency = _clip(rng.gauss(lat_mean, lat_sd), 0.2)
        iterations = int(_clip(round(rng.gauss(iter_mean, 1.0)), 0, 6))
        # Drawn independently of quality/latency/iterations/status -- a decomposition
        # length correlated with the archetype's task category is a legitimate,
        # pre-decision property, not a proxy for this record's outcome.
        plan_steps = int(_clip(round(rng.gauss(plan_steps_mean, 1.0)), 0, 6))
        status = "completed" if rng.random() > 0.08 else "failed"
        timestamp = timestamp + timedelta(minutes=rng.randint(1, 30))

        repository.add(
            ExperienceRecord(
                experience_id=f"exp-{i:04d}",
                session_id=f"session-{i // 10:03d}",
                task_id=f"task-{i:04d}",
                timestamp=timestamp,
                state_features={
                    "task_type": task_type,
                    "max_iterations": DEFAULT_MAX_ITERATIONS,
                    "planner_output": {
                        "decomposition": [f"step-{s}" for s in range(plan_steps)]
                    },
                    "error_feature_count": 0,
                    "worker_output_count": len(critics),
                },
                selected_critics=critics,
                critic_scores={
                    critic: _clip(quality + rng.gauss(0.0, 0.05), 0.0, 1.0) for critic in critics
                },
                aggregated_quality_score=quality,
                iterations=iterations,
                execution_status=status,
                latency=latency,
            )
        )
    return repository


# --- 2. Small aggregation glue, mirroring OfflineEvaluator/ExperimentRunner's own formulas ---
#
# `OfflineEvaluator.evaluate` and `ExperimentRunner._aggregate` are hardcoded/private
# and cannot aggregate a `replay_with_learning()` result (ExperimentRunner never calls
# that method). Rather than reach into either's private internals, these two small
# functions mirror their public-facing formulas exactly, using only public building
# blocks (`app.evaluation.experiments.Analyzer`, plain arithmetic) -- no replay,
# resampling, or statistical-test logic is reimplemented; only the "shape a list of
# ReplayStep/ReplayResult into the next model up" glue is duplicated, exactly as
# `app/evaluation/ablation/runner.py::_experiment_result_to_replay_result` already does
# for a parallel reason.


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate_replay_steps(policy_name: str, steps: list[ReplayStep], total_stored: int):
    from app.evaluation.offline.models import ReplayResult

    total_experiences = len(steps)
    rewards = [step.reward for step in steps]
    qualities = [step.quality if step.quality is not None else 0.0 for step in steps]
    iterations = [float(step.iterations) for step in steps]
    latencies = [step.latency if step.latency is not None else 0.0 for step in steps]

    total_reward = sum(rewards)
    frequency_counts: dict[str, int] = {}
    for step in steps:
        for critic_name in step.selected_critics:
            frequency_counts[critic_name] = frequency_counts.get(critic_name, 0) + 1
    critic_selection_frequency = (
        {name: count / total_experiences for name, count in frequency_counts.items()}
        if total_experiences
        else {}
    )

    return ReplayResult(
        policy_name=policy_name,
        total_experiences=total_experiences,
        total_reward=total_reward,
        average_reward=total_reward / total_experiences if total_experiences else 0.0,
        average_quality=_mean(qualities),
        average_iterations=_mean(iterations),
        average_latency=_mean(latencies),
        critic_selection_frequency=critic_selection_frequency,
        metadata={
            "total_stored_experiences": total_stored,
            "match_rate": total_experiences / total_stored if total_stored else 0.0,
        },
    )


def _aggregate_experiment_result(
    experiment_name: str,
    policy_name: str,
    runs: list,
    random_seed: int,
    num_runs: int,
) -> ExperimentResult:
    from app.evaluation.experiments import Analyzer as ExperimentAnalyzer

    analyzer = ExperimentAnalyzer()
    reward_summary = analyzer.summarize([run.average_reward for run in runs])
    quality_values = [run.average_quality for run in runs]
    latency_values = [run.average_latency for run in runs]
    iteration_values = [run.average_iterations for run in runs]
    match_rate_values = [float(run.metadata.get("match_rate", 0.0)) for run in runs]

    critic_names = sorted({name for run in runs for name in run.critic_selection_frequency})
    critic_selection_frequency = {
        name: analyzer.mean([run.critic_selection_frequency.get(name, 0.0) for run in runs])
        for name in critic_names
    }

    return ExperimentResult(
        experiment_name=experiment_name,
        policy_name=policy_name,
        runs=runs,
        average_reward=reward_summary.mean,
        std_reward=reward_summary.std_dev,
        average_quality=analyzer.mean(quality_values),
        average_latency=analyzer.mean(latency_values),
        average_iterations=analyzer.mean(iteration_values),
        match_rate=analyzer.mean(match_rate_values),
        critic_selection_frequency=critic_selection_frequency,
        metadata={
            "random_seed": random_seed,
            "num_runs": num_runs,
            "reward_confidence_interval_95": {
                "lower": reward_summary.confidence_interval.lower,
                "upper": reward_summary.confidence_interval.upper,
            },
            "reward_minimum": reward_summary.minimum,
            "reward_maximum": reward_summary.maximum,
        },
    )


# --- 3. Core bootstrap-based experiments (Heuristic baseline, Cold-Start LinUCB sweep) ---


def run_core_bootstrap_experiments(repository: InMemoryExperienceRepository) -> dict:
    runner = ExperimentRunner(repository=repository)
    results = {}

    results["heuristic"] = runner.run(
        ExperimentConfig(
            experiment_name="1-heuristic-baseline",
            policy_name="HeuristicPolicy",
            random_seed=CAMPAIGN_SEED,
            num_runs=NUM_RUNS,
            candidate_actions=CANDIDATE_ACTIONS,
        )
    )
    for alpha in ALPHAS:
        results[f"cold_start_alpha_{alpha}"] = runner.run(
            ExperimentConfig(
                experiment_name=f"2-cold-start-linucb-alpha-{alpha}",
                policy_name="LinUCBPolicy",
                alpha=alpha,
                random_seed=CAMPAIGN_SEED,
                num_runs=NUM_RUNS,
                candidate_actions=CANDIDATE_ACTIONS,
            )
        )
    return results


# --- 4. Sequential Learning LinUCB: bootstrap statistics, via replay_with_learning ---


def run_sequential_learning_bootstrap(
    repository: InMemoryExperienceRepository, alpha: float
) -> ExperimentResult:
    """Bootstrap-resample `repository` NUM_RUNS times; within each run, train a
    *fresh* LinUCBPolicy sequentially (via `ReplayEngine.replay_with_learning`)
    over that run's resample, then aggregate -- mirroring exactly what
    `ExperimentRunner.run` does for `replay()`, since `ExperimentRunner` itself
    has no path to `replay_with_learning`.
    """
    source_records = repository.list()
    reward_calculator = RewardCalculator()
    rng = random.Random(CAMPAIGN_SEED)

    run_results = []
    for _ in range(NUM_RUNS):
        resampled_records = (
            [rng.choice(source_records) for _ in range(len(source_records))]
            if source_records
            else []
        )
        resampled_repo = BootstrapExperienceRepository(resampled_records)
        policy = LinUCBPolicy(alpha=alpha)
        engine = ReplayEngine(
            repository=resampled_repo,
            policy=policy,
            reward_calculator=reward_calculator,
            candidate_actions=CANDIDATE_ACTIONS,
        )
        steps = engine.replay_with_learning()
        run_results.append(
            _aggregate_replay_steps("LinUCBPolicy", steps, resampled_repo.count())
        )

    return _aggregate_experiment_result(
        experiment_name=f"3-sequential-learning-linucb-alpha-{alpha}",
        policy_name="LinUCBPolicy",
        runs=run_results,
        random_seed=CAMPAIGN_SEED,
        num_runs=NUM_RUNS,
    )


def run_sequential_learning_experiments(repository: InMemoryExperienceRepository) -> dict:
    return {
        f"sequential_alpha_{alpha}": run_sequential_learning_bootstrap(repository, alpha)
        for alpha in ALPHAS
    }


# --- 5. Ablations (Random Critic, Reduced Context, Quality-only Reward) ---


def run_ablations(repository: InMemoryExperienceRepository) -> dict:
    ablation_runner = AblationRunner(repository=repository)
    configs = {
        "random_critic": AblationConfig(
            experiment_name="4-ablation-random-critic",
            baseline_policy="LinUCBPolicy",
            candidate_policy="RandomCriticPolicy",
            ablation_type="random_critic_selection",
        ),
        "reduced_context": AblationConfig(
            experiment_name="5-ablation-reduced-context",
            baseline_policy="LinUCBPolicy",
            candidate_policy="LinUCBPolicy",
            ablation_type="reduced_context_features",
        ),
        "quality_only_reward": AblationConfig(
            experiment_name="6-ablation-quality-only-reward",
            baseline_policy="LinUCBPolicy",
            candidate_policy="LinUCBPolicy",
            ablation_type="alternative_reward_definitions",
        ),
    }
    return {
        key: ablation_runner.run(
            config,
            num_runs=NUM_RUNS,
            random_seed=CAMPAIGN_SEED,
            candidate_actions=CANDIDATE_ACTIONS,
        )
        for key, config in configs.items()
    }


def replay_ablation_candidate_arms(repository: InMemoryExperienceRepository) -> dict:
    """Independently replay each ablation's candidate arm (mirrors AblationRunner's
    own internal wiring, using its exported policy/strategy classes) to obtain a
    full ExperimentResult -- see campaign_v1's identical rationale for why
    AblationRunner.run() alone doesn't expose this.
    """

    def random_critic_factory(config: ExperimentConfig):
        return RandomCriticPolicy(seed=config.random_seed)

    def reduced_context_factory(config: ExperimentConfig):
        return ReducedContextPolicy(LinUCBPolicy(alpha=config.alpha or 1.0), keep_fraction=0.5)

    results = {}

    random_critic_runner = ExperimentRunner(
        repository=repository, policy_factory=random_critic_factory
    )
    results["random_critic"] = random_critic_runner.run(
        ExperimentConfig(
            experiment_name="4-candidate-random-critic",
            policy_name="RandomCriticPolicy",
            random_seed=CAMPAIGN_SEED,
            num_runs=NUM_RUNS,
            candidate_actions=CANDIDATE_ACTIONS,
        )
    )

    reduced_context_runner = ExperimentRunner(
        repository=repository, policy_factory=reduced_context_factory
    )
    results["reduced_context"] = reduced_context_runner.run(
        ExperimentConfig(
            experiment_name="5-candidate-reduced-context",
            policy_name="LinUCBPolicy",
            alpha=1.0,
            random_seed=CAMPAIGN_SEED,
            num_runs=NUM_RUNS,
            candidate_actions=CANDIDATE_ACTIONS,
        )
    )

    quality_only_runner = ExperimentRunner(
        repository=repository,
        reward_calculator=RewardCalculator(strategy=QualityOnlyRewardStrategy()),
    )
    results["quality_only_reward"] = quality_only_runner.run(
        ExperimentConfig(
            experiment_name="6-candidate-quality-only-reward",
            policy_name="LinUCBPolicy",
            alpha=1.0,
            random_seed=CAMPAIGN_SEED,
            num_runs=NUM_RUNS,
            candidate_actions=CANDIDATE_ACTIONS,
        )
    )
    return results


# --- 6. Raw single-pass ReplayEngine + LearningAnalyzer: reward-per-step, cumulative
#        reward/regret, moving average, convergence, learning rate -- for every
#        experiment, over the FULL (non-resampled) repository, exactly once. ---


def raw_replay_curve(
    repository: InMemoryExperienceRepository, policy, reward_calculator=None
):
    """Run one deterministic `.replay()` pass and analyze it with `LearningAnalyzer`.

    Used for every non-training arm (Heuristic, Cold-Start LinUCB, Random
    Critic, Reduced Context, Quality-only Reward candidate/baseline).
    """
    engine = ReplayEngine(
        repository=repository,
        policy=policy,
        reward_calculator=reward_calculator or RewardCalculator(),
        candidate_actions=CANDIDATE_ACTIONS,
    )
    steps = engine.replay()
    return LearningAnalyzer().analyze(steps)


def raw_sequential_learning_curve(repository: InMemoryExperienceRepository, alpha: float):
    """Run one deterministic `.replay_with_learning()` pass and analyze it.

    This is the canonical Sequential Learning LinUCB curve at `alpha`: a
    single fresh `LinUCBPolicy`, trained sequentially over the full,
    non-resampled repository, exactly once (no bootstrap resampling --
    mirrors how `ExperimentRunner.run` treats `num_runs == 1`).
    """
    policy = LinUCBPolicy(alpha=alpha)
    engine = ReplayEngine(
        repository=repository,
        policy=policy,
        reward_calculator=RewardCalculator(),
        candidate_actions=CANDIDATE_ACTIONS,
    )
    steps = engine.replay_with_learning()
    return LearningAnalyzer().analyze(steps)


def build_raw_curves(repository: InMemoryExperienceRepository) -> dict:
    curves = {
        "heuristic": raw_replay_curve(repository, HeuristicPolicy()),
    }
    for alpha in ALPHAS:
        curves[f"cold_start_alpha_{alpha}"] = raw_replay_curve(
            repository, LinUCBPolicy(alpha=alpha)
        )
        curves[f"sequential_alpha_{alpha}"] = raw_sequential_learning_curve(repository, alpha)

    curves["random_critic"] = raw_replay_curve(
        repository, RandomCriticPolicy(seed=CAMPAIGN_SEED)
    )
    curves["reduced_context"] = raw_replay_curve(
        repository, ReducedContextPolicy(LinUCBPolicy(alpha=1.0), keep_fraction=0.5)
    )
    curves["quality_only_reward"] = raw_replay_curve(
        repository, LinUCBPolicy(alpha=1.0), RewardCalculator(strategy=QualityOnlyRewardStrategy())
    )
    return curves


# --- 7. Statistical comparisons (app.evaluation.statistics.Analyzer, unmodified) ---


def compute_statistics(
    core_results: dict, sequential_results: dict, ablation_candidates: dict
) -> dict:
    from app.evaluation.statistics import Analyzer as StatisticsAnalyzer

    analyzer = StatisticsAnalyzer()
    baseline = core_results["heuristic"]
    full_acrf = core_results["cold_start_alpha_1.0"]
    comparisons = {}

    # vs Heuristic baseline (Table 1)
    for key, result in core_results.items():
        if key == "heuristic":
            continue
        comparisons[f"{key}_vs_heuristic"] = analyzer.compare_experiments(
            baseline, result, metric="average_reward"
        )
    for key, result in sequential_results.items():
        comparisons[f"{key}_vs_heuristic"] = analyzer.compare_experiments(
            baseline, result, metric="average_reward"
        )

    # Sequential Learning vs Cold-Start, same alpha (Table 3 -- the key comparison)
    for alpha in ALPHAS:
        comparisons[f"sequential_vs_cold_start_alpha_{alpha}"] = analyzer.compare_experiments(
            core_results[f"cold_start_alpha_{alpha}"],
            sequential_results[f"sequential_alpha_{alpha}"],
            metric="average_reward",
        )

    # Ablations vs Full ACRF (Table 2)
    for key, result in ablation_candidates.items():
        comparisons[f"{key}_vs_full_acrf"] = analyzer.compare_experiments(
            full_acrf, result, metric="average_reward"
        )

    return comparisons


# --- 8. Tables ---


def _fmt_ci(ci: dict) -> str:
    return f"[{ci['lower']:.4f}, {ci['upper']:.4f}]"


def _fmt_p(p_value: float) -> str:
    return f"{p_value:.4f}" if p_value >= 0.0001 else "<0.0001"


def build_table1(core_results: dict, sequential_results: dict, comparisons: dict) -> str:
    """Table 1: Overall comparison of all policies."""
    header = (
        "| Policy | Avg Reward | Avg Quality | Avg Latency | Avg Iterations | "
        "Match Rate | 95% CI (reward) | p-value | Effect Size | Significant |"
    )
    separator = "|---|---|---|---|---|---|---|---|---|---|"
    rows = [header, separator]

    def _row(label: str, result: ExperimentResult, comparison_key: str | None) -> str:
        ci = _fmt_ci(result.metadata["reward_confidence_interval_95"])
        if comparison_key is None:
            p_value, effect, sig = "n/a", "-", "-"
        else:
            comparison = comparisons[comparison_key]
            p_value, effect, sig = (
                _fmt_p(comparison.p_value),
                f"{comparison.effect_size:.4f}",
                comparison.significant,
            )
        return (
            f"| {label} | {result.average_reward:.4f} | {result.average_quality:.4f} | "
            f"{result.average_latency:.4f} | {result.average_iterations:.4f} | "
            f"{result.match_rate:.4f} | {ci} | {p_value} | {effect} | {sig} |"
        )

    rows.append(_row("1. Heuristic Policy (Baseline)", core_results["heuristic"], None))
    for alpha in ALPHAS:
        rows.append(
            _row(
                f"2. Cold-Start LinUCB (alpha={alpha})",
                core_results[f"cold_start_alpha_{alpha}"],
                f"cold_start_alpha_{alpha}_vs_heuristic",
            )
        )
    rows.append(
        _row(
            f"3. Sequential Learning LinUCB (alpha={CANONICAL_ALPHA}, canonical)",
            sequential_results[f"sequential_alpha_{CANONICAL_ALPHA}"],
            f"sequential_alpha_{CANONICAL_ALPHA}_vs_heuristic",
        )
    )
    return "\n".join(rows) + "\n"


def build_table2(ablation_results: dict, ablation_candidates: dict) -> str:
    """Table 2: Ablation study (each candidate vs Full ACRF)."""
    labels = {
        "random_critic": "4. Random Critic",
        "reduced_context": "5. Reduced Context Ablation",
        "quality_only_reward": "6. Quality-only Reward Ablation",
    }
    header = (
        "| Ablation | Baseline Reward (Full ACRF) | Candidate Reward | Reward Diff | "
        "Quality Diff | Latency Diff | Iteration Diff | Match Rate | Winner | p-value | "
        "Effect Size | Significant |"
    )
    separator = "|---|---|---|---|---|---|---|---|---|---|---|---|"
    rows = [header, separator]
    for key, label in labels.items():
        result = ablation_results[key]
        candidate = ablation_candidates[key]
        rows.append(
            f"| {label} | {result.baseline_reward:.4f} | {result.candidate_reward:.4f} | "
            f"{result.reward_difference:+.4f} | {result.quality_difference:+.4f} | "
            f"{result.latency_difference:+.4f} | {result.iteration_difference:+.4f} | "
            f"{candidate.match_rate:.4f} | {result.metadata['winner']} | "
            f"{_fmt_p(result.metadata['p_value'])} | {result.metadata['effect_size']:.4f} | "
            f"{result.metadata['significant']} |"
        )
    return "\n".join(rows) + "\n"


def build_table3(
    core_results: dict, sequential_results: dict, comparisons: dict, raw_curves: dict
) -> str:
    """Table 3: Sequential Learning analysis (Cold-Start vs Sequential, same alpha)."""
    header = (
        "| Alpha | Cold-Start Reward | Sequential Reward | Reward Diff | p-value | "
        "Effect Size | Significant | Convergence Step | Learning Rate | Final Cum. Regret |"
    )
    separator = "|---|---|---|---|---|---|---|---|---|---|"
    rows = [header, separator]
    for alpha in ALPHAS:
        cold = core_results[f"cold_start_alpha_{alpha}"]
        sequential = sequential_results[f"sequential_alpha_{alpha}"]
        comparison = comparisons[f"sequential_vs_cold_start_alpha_{alpha}"]
        curve = raw_curves[f"sequential_alpha_{alpha}"]
        convergence = curve.metadata.get("convergence_point")
        rate = curve.metadata.get("learning_rate_estimate", 0.0)
        final_regret = curve.cumulative_regret[-1] if curve.cumulative_regret else 0.0
        rows.append(
            f"| {alpha} | {cold.average_reward:.4f} | {sequential.average_reward:.4f} | "
            f"{comparison.mean_difference:+.4f} | {_fmt_p(comparison.p_value)} | "
            f"{comparison.effect_size:.4f} | {comparison.significant} | "
            f"{convergence if convergence is not None else 'n/a'} | {rate:+.4f} | "
            f"{final_regret:.4f} |"
        )
    return "\n".join(rows) + "\n"


def build_table4(raw_curves: dict) -> str:
    """Table 4: Learning Analysis summary, over every experiment's raw (single-pass) curve."""
    labels = {
        "heuristic": "1. Heuristic Policy",
        **{f"cold_start_alpha_{a}": f"2. Cold-Start LinUCB (alpha={a})" for a in ALPHAS},
        **{f"sequential_alpha_{a}": f"3. Sequential Learning LinUCB (alpha={a})" for a in ALPHAS},
        "random_critic": "4. Random Critic",
        "reduced_context": "5. Reduced Context",
        "quality_only_reward": "6. Quality-only Reward",
    }
    header = (
        "| Experiment | Steps Matched | Avg Reward | Final Cum. Reward | Final Cum. Regret | "
        "Convergence Step | Learning Rate |"
    )
    separator = "|---|---|---|---|---|---|---|"
    rows = [header, separator]
    for key, label in labels.items():
        curve = raw_curves[key]
        n = curve.metadata.get("num_steps", 0)
        final_cum_reward = curve.cumulative_reward[-1] if curve.cumulative_reward else 0.0
        final_cum_regret = curve.cumulative_regret[-1] if curve.cumulative_regret else 0.0
        convergence = curve.metadata.get("convergence_point")
        rate = curve.metadata.get("learning_rate_estimate", 0.0)
        rows.append(
            f"| {label} | {n} | {curve.average_reward:.4f} | {final_cum_reward:.4f} | "
            f"{final_cum_regret:.4f} | {convergence if convergence is not None else 'n/a'} | "
            f"{rate:+.4f} |"
        )
    return "\n".join(rows) + "\n"


# --- 9. Figures ---

_KEY_SERIES_LABELS = {
    "heuristic": "Heuristic Policy",
    f"cold_start_alpha_{CANONICAL_ALPHA}": "Cold-Start LinUCB",
    f"sequential_alpha_{CANONICAL_ALPHA}": "Sequential Learning LinUCB",
    "random_critic": "Random Critic",
}


def generate_figures(
    raw_curves: dict, core_results: dict, sequential_results: dict, ablation_candidates: dict
) -> None:
    plt.rcParams["figure.autolayout"] = True

    def _key_curves():
        return {label: raw_curves[key] for key, label in _KEY_SERIES_LABELS.items()}

    # 1. Reward Curve
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, curve in _key_curves().items():
        ax.plot(
            range(1, len(curve.reward_per_step) + 1),
            curve.reward_per_step,
            marker=".",
            label=label,
        )
    ax.set_xlabel("Step (matched experience index)")
    ax.set_ylabel("Reward")
    ax.set_title("Reward Curve")
    ax.legend(fontsize=8)
    fig.savefig(FIGURES_DIR / "reward_curve.png", dpi=150)
    plt.close(fig)

    # 2. Cumulative Reward
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, curve in _key_curves().items():
        ax.plot(range(1, len(curve.cumulative_reward) + 1), curve.cumulative_reward, label=label)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative reward")
    ax.set_title("Cumulative Reward")
    ax.legend(fontsize=8)
    fig.savefig(FIGURES_DIR / "cumulative_reward.png", dpi=150)
    plt.close(fig)

    # 3. Moving Average Reward
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, curve in _key_curves().items():
        ax.plot(
            range(1, len(curve.moving_average_reward) + 1), curve.moving_average_reward, label=label
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Moving average reward (window=10)")
    ax.set_title("Moving Average Reward")
    ax.legend(fontsize=8)
    fig.savefig(FIGURES_DIR / "moving_average_reward.png", dpi=150)
    plt.close(fig)

    # 4. Instantaneous Regret
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, curve in _key_curves().items():
        ax.plot(
            range(1, len(curve.instantaneous_regret) + 1),
            curve.instantaneous_regret,
            marker=".",
            label=label,
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Instantaneous regret")
    ax.set_title("Instantaneous Regret")
    ax.legend(fontsize=8)
    fig.savefig(FIGURES_DIR / "instantaneous_regret.png", dpi=150)
    plt.close(fig)

    # 5. Cumulative Regret
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, curve in _key_curves().items():
        ax.plot(range(1, len(curve.cumulative_regret) + 1), curve.cumulative_regret, label=label)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative regret")
    ax.set_title("Cumulative Regret")
    ax.legend(fontsize=8)
    fig.savefig(FIGURES_DIR / "cumulative_regret.png", dpi=150)
    plt.close(fig)

    # 6. Critic Selection Distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    freq_sources = {
        "Heuristic Policy": core_results["heuristic"],
        "Cold-Start LinUCB": core_results[f"cold_start_alpha_{CANONICAL_ALPHA}"],
        "Sequential Learning LinUCB": sequential_results[f"sequential_alpha_{CANONICAL_ALPHA}"],
        "Random Critic": ablation_candidates["random_critic"],
    }
    critics = CANDIDATE_ACTIONS
    width = 0.2
    x = range(len(critics))
    for offset, (label, result) in zip((-1.5, -0.5, 0.5, 1.5), freq_sources.items(), strict=True):
        heights = [result.critic_selection_frequency.get(critic, 0.0) for critic in critics]
        ax.bar([xi + offset * width for xi in x], heights, width=width, label=label)
    ax.set_xticks(list(x))
    ax.set_xticklabels(critics)
    ax.set_ylabel("Selection frequency")
    ax.set_title("Critic Selection Distribution")
    ax.legend(fontsize=8)
    fig.savefig(FIGURES_DIR / "critic_selection_distribution.png", dpi=150)
    plt.close(fig)

    # 7. Latency Comparison (across every bootstrap-evaluated experiment)
    fig, ax = plt.subplots(figsize=(11, 5))
    latency_labels = (
        ["Heuristic"]
        + [f"Cold-Start\nalpha={a}" for a in ALPHAS]
        + [f"Sequential\nalpha={a}" for a in ALPHAS]
        + ["Random\nCritic", "Reduced\nContext", "Quality-only\nReward"]
    )
    latency_values = (
        [core_results["heuristic"].average_latency]
        + [core_results[f"cold_start_alpha_{a}"].average_latency for a in ALPHAS]
        + [sequential_results[f"sequential_alpha_{a}"].average_latency for a in ALPHAS]
        + [
            ablation_candidates["random_critic"].average_latency,
            ablation_candidates["reduced_context"].average_latency,
            ablation_candidates["quality_only_reward"].average_latency,
        ]
    )
    ax.bar(range(len(latency_labels)), latency_values, color="tab:orange")
    ax.set_xticks(range(len(latency_labels)))
    ax.set_xticklabels(latency_labels, fontsize=7)
    ax.set_ylabel("Average latency")
    ax.set_title("Latency Comparison")
    fig.savefig(FIGURES_DIR / "latency_comparison.png", dpi=150)
    plt.close(fig)

    # 8. Reward Distribution (per-run average_reward across the NUM_RUNS bootstrap runs)
    fig, ax = plt.subplots(figsize=(9, 5))
    distribution_sources = {
        "Heuristic": core_results["heuristic"],
        "Cold-Start\nLinUCB": core_results[f"cold_start_alpha_{CANONICAL_ALPHA}"],
        "Sequential\nLearning": sequential_results[f"sequential_alpha_{CANONICAL_ALPHA}"],
        "Random\nCritic": ablation_candidates["random_critic"],
    }
    data = [[run.average_reward for run in result.runs] for result in distribution_sources.values()]
    ax.boxplot(data, tick_labels=list(distribution_sources.keys()))
    ax.set_ylabel(f"Average reward per bootstrap run (n={NUM_RUNS})")
    ax.set_title("Reward Distribution Across Bootstrap Runs")
    fig.savefig(FIGURES_DIR / "reward_distribution.png", dpi=150)
    plt.close(fig)

    # 9. Convergence Analysis
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for (label, curve), color in zip(_key_curves().items(), colors, strict=False):
        moving_avg = curve.moving_average_reward
        ax.plot(range(1, len(moving_avg) + 1), moving_avg, label=label, color=color)
        convergence = curve.metadata.get("convergence_point")
        if convergence is not None and moving_avg:
            ax.axvline(convergence + 1, color=color, linestyle="--", alpha=0.6)
    ax.set_xlabel("Step")
    ax.set_ylabel("Moving average reward")
    ax.set_title("Convergence Analysis (dashed lines mark each series' convergence step)")
    ax.legend(fontsize=8)
    fig.savefig(FIGURES_DIR / "convergence_analysis.png", dpi=150)
    plt.close(fig)


# --- 10. Main ---

if __name__ == "__main__":
    start_time = time.perf_counter()

    print(f"Generating synthetic experience log (seed={DATA_SEED}, n={NUM_RECORDS})...")
    repository = generate_repository(DATA_SEED, NUM_RECORDS)
    dumps_before = {record.experience_id: record.model_dump() for record in repository.list()}
    print(f"Repository size: {repository.count()}")

    print("Running core bootstrap experiments (Heuristic baseline, Cold-Start LinUCB sweep)...")
    core_results = run_core_bootstrap_experiments(repository)

    print("Running Sequential Learning LinUCB (bootstrap, via replay_with_learning)...")
    sequential_results = run_sequential_learning_experiments(repository)

    print("Running ablations (Random Critic, Reduced Context, Quality-only Reward)...")
    ablation_results = run_ablations(repository)
    ablation_candidates = replay_ablation_candidate_arms(repository)

    print("Computing raw single-pass replay curves (LearningAnalyzer) for every experiment...")
    raw_curves = build_raw_curves(repository)

    print("Computing statistical comparisons...")
    comparisons = compute_statistics(core_results, sequential_results, ablation_candidates)

    print(f"Verifying source repository unmutated (must equal {NUM_RECORDS} records)...")
    assert repository.count() == NUM_RECORDS, "source repository was mutated during the campaign"
    for record in repository.list():
        assert record.model_dump() == dumps_before[record.experience_id], (
            f"record {record.experience_id} was mutated during the campaign"
        )

    print("Building tables...")
    table1 = build_table1(core_results, sequential_results, comparisons)
    table2 = build_table2(ablation_results, ablation_candidates)
    table3 = build_table3(core_results, sequential_results, comparisons, raw_curves)
    table4 = build_table4(raw_curves)
    (RESULTS_DIR / "table1_overall_comparison.md").write_text(table1, encoding="utf-8")
    (RESULTS_DIR / "table2_ablation_study.md").write_text(table2, encoding="utf-8")
    (RESULTS_DIR / "table3_sequential_learning_analysis.md").write_text(table3, encoding="utf-8")
    (RESULTS_DIR / "table4_learning_analysis_summary.md").write_text(table4, encoding="utf-8")

    print("Generating figures...")
    generate_figures(raw_curves, core_results, sequential_results, ablation_candidates)

    print("Exporting raw results (JSON/CSV/Markdown) via existing exporters...")
    experiment_exporter = ExperimentExporter()
    all_core_and_sequential = list(core_results.values()) + list(sequential_results.values())
    (RESULTS_DIR / "experiments.json").write_text(
        experiment_exporter.to_json(all_core_and_sequential), encoding="utf-8"
    )
    (RESULTS_DIR / "experiments.csv").write_text(
        experiment_exporter.to_csv(all_core_and_sequential), encoding="utf-8"
    )
    (RESULTS_DIR / "experiments_summary.md").write_text(
        experiment_exporter.to_markdown(all_core_and_sequential), encoding="utf-8"
    )

    ablation_report_generator = AblationReportGenerator()
    all_ablation_results = list(ablation_results.values())
    (RESULTS_DIR / "ablations.json").write_text(
        ablation_report_generator.to_json(all_ablation_results), encoding="utf-8"
    )
    (RESULTS_DIR / "ablations.csv").write_text(
        ablation_report_generator.to_csv(all_ablation_results), encoding="utf-8"
    )
    (RESULTS_DIR / "ablations_report.md").write_text(
        ablation_report_generator.to_markdown(all_ablation_results), encoding="utf-8"
    )

    learning_report_generator = LearningReportGenerator()
    for key in (
        "heuristic",
        f"cold_start_alpha_{CANONICAL_ALPHA}",
        f"sequential_alpha_{CANONICAL_ALPHA}",
        "random_critic",
    ):
        curve = raw_curves[key]
        safe_key = key.replace(".", "_")
        (RESULTS_DIR / f"learning_curve_{safe_key}.csv").write_text(
            learning_report_generator.to_csv(curve), encoding="utf-8"
        )
        (RESULTS_DIR / f"learning_curve_{safe_key}.json").write_text(
            learning_report_generator.to_json(curve), encoding="utf-8"
        )
    (RESULTS_DIR / "learning_curves_all.json").write_text(
        json.dumps(
            {key: curve.model_dump(mode="json") for key, curve in raw_curves.items()}, indent=2
        ),
        encoding="utf-8",
    )

    elapsed = time.perf_counter() - start_time
    print(f"Campaign complete in {elapsed:.2f}s. Results written to: {RESULTS_DIR}")
    print(f"Figures written to: {FIGURES_DIR}")
    (RESULTS_DIR / "runtime.txt").write_text(
        f"Total campaign runtime: {elapsed:.2f} seconds\n", encoding="utf-8"
    )
