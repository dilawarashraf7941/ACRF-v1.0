"""ACRF v1.0 Experimental Campaign — execution script.

Runs the campaign specified for this session: 1 baseline (HeuristicPolicy),
5 LinUCB alpha candidates (0, 0.25, 0.5, 1.0, 2.0), 1 "Full ACRF" reference
run (LinUCB alpha=1.0), and 3 ablations (Random Critic, Reduced Context,
Quality-only Reward).

This script is execution-only: it calls nothing but already-existing,
unmodified ACRF evaluation classes (`ExperienceRecord`,
`InMemoryExperienceRepository`, `ExperimentConfig`, `ExperimentRunner`,
`AblationConfig`, `AblationRunner`, `app.evaluation.statistics.Analyzer`,
`app.evaluation.experiments.Exporter`,
`app.evaluation.ablation.AblationReportGenerator`). No new framework code,
no new package under `app/`, no modification to any existing module.

Reproducibility: the synthetic experience log is generated with
DATA_SEED; every experiment/ablation run uses CAMPAIGN_SEED. Both are
fixed constants below, so re-running this script reproduces every number
in Results.md exactly.
"""

import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.evaluation.ablation import (
    AblationConfig,
    AblationReportGenerator,
    AblationRunner,
    QualityOnlyRewardStrategy,
    RandomCriticPolicy,
    ReducedContextPolicy,
)
from app.evaluation.experiments import ExperimentConfig, ExperimentRunner
from app.evaluation.experiments import Exporter as ExperimentExporter
from app.evaluation.statistics import Analyzer as StatisticsAnalyzer
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
ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0]
CANDIDATE_ACTIONS = ["LogicCritic", "CodeCritic", "FactCritic", "MetaCritic"]

# --- 1. Synthetic experience log ---
# No live LLM/agent execution exists in this framework by design (see
# app/evaluation/offline/README.md); ExperienceRecord instances must be
# supplied from somewhere. This generator produces a plausible,
# deterministic historical log with task-archetype-correlated
# quality/latency/iteration signals, documented in full in
# Threats_to_Validity.md.

_ARCHETYPES: list[tuple[float, list[str] | None, float, float, float, float, float]] = [
    # (weight, selected_critics, quality_mean, quality_sd, latency_mean, latency_sd, iterations_mean)
    (0.40, ["CodeCritic"], 0.78, 0.08, 1.1, 0.25, 0.8),
    (0.20, ["LogicCritic"], 0.68, 0.09, 1.4, 0.30, 1.3),
    (0.15, ["FactCritic"], 0.60, 0.10, 1.6, 0.30, 1.6),
    (0.10, ["MetaCritic"], 0.58, 0.12, 1.8, 0.35, 2.0),
    (0.15, None, 0.50, 0.13, 2.2, 0.40, 2.8),  # multi-critic episodes
]


def _clip(value: float, low: float, high: float | None = None) -> float:
    value = max(low, value)
    return min(value, high) if high is not None else value


def generate_repository(seed: int, n: int) -> InMemoryExperienceRepository:
    rng = random.Random(seed)
    repository = InMemoryExperienceRepository()
    weights = [archetype[0] for archetype in _ARCHETYPES]
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)

    for i in range(n):
        weight, critics, q_mean, q_sd, lat_mean, lat_sd, iter_mean = rng.choices(
            _ARCHETYPES, weights=weights, k=1
        )[0]
        if critics is None:
            critics = rng.choice([["FactCritic", "MetaCritic"], ["LogicCritic", "MetaCritic"]])

        quality = _clip(rng.gauss(q_mean, q_sd), 0.0, 1.0)
        latency = _clip(rng.gauss(lat_mean, lat_sd), 0.2)
        iterations = int(_clip(round(rng.gauss(iter_mean, 1.0)), 0, 6))
        status = "completed" if rng.random() > 0.08 else "failed"
        timestamp = timestamp + timedelta(minutes=rng.randint(1, 30))

        repository.add(
            ExperienceRecord(
                experience_id=f"exp-{i:04d}",
                session_id=f"session-{i // 10:03d}",
                task_id=f"task-{i:04d}",
                timestamp=timestamp,
                state_features={"error_feature_count": 0, "worker_output_count": len(critics)},
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


# --- 2. Core experiments (1 baseline + 5 alpha candidates + Full ACRF reference) ---


def run_core_experiments(repository: InMemoryExperienceRepository) -> dict:
    runner = ExperimentRunner(repository=repository)
    results = {}

    results["baseline"] = runner.run(
        ExperimentConfig(
            experiment_name="1-baseline-heuristic",
            policy_name="HeuristicPolicy",
            random_seed=CAMPAIGN_SEED,
            num_runs=NUM_RUNS,
            candidate_actions=CANDIDATE_ACTIONS,
        )
    )

    for index, alpha in enumerate(ALPHAS, start=2):
        results[f"linucb_alpha_{alpha}"] = runner.run(
            ExperimentConfig(
                experiment_name=f"{index}-linucb-alpha-{alpha}",
                policy_name="LinUCBPolicy",
                alpha=alpha,
                random_seed=CAMPAIGN_SEED,
                num_runs=NUM_RUNS,
                candidate_actions=CANDIDATE_ACTIONS,
            )
        )

    results["full_acrf"] = runner.run(
        ExperimentConfig(
            experiment_name="10-full-acrf",
            policy_name="LinUCBPolicy",
            alpha=1.0,
            random_seed=CAMPAIGN_SEED,
            num_runs=NUM_RUNS,
            candidate_actions=CANDIDATE_ACTIONS,
        )
    )
    return results


# --- 3. Ablations (Random Critic, Reduced Context, Quality-only Reward) ---


def run_ablations(repository: InMemoryExperienceRepository) -> dict:
    ablation_runner = AblationRunner(repository=repository)
    configs = {
        "random_critic": AblationConfig(
            experiment_name="7-ablation-random-critic",
            baseline_policy="LinUCBPolicy",
            candidate_policy="RandomCriticPolicy",
            ablation_type="random_critic_selection",
        ),
        "reduced_context": AblationConfig(
            experiment_name="8-ablation-reduced-context",
            baseline_policy="LinUCBPolicy",
            candidate_policy="LinUCBPolicy",
            ablation_type="reduced_context_features",
        ),
        "quality_only_reward": AblationConfig(
            experiment_name="9-ablation-quality-only-reward",
            baseline_policy="LinUCBPolicy",
            candidate_policy="LinUCBPolicy",
            ablation_type="alternative_reward_definitions",
        ),
    }
    return {
        key: ablation_runner.run(
            config, num_runs=NUM_RUNS, random_seed=CAMPAIGN_SEED, candidate_actions=CANDIDATE_ACTIONS
        )
        for key, config in configs.items()
    }


# --- 4. Statistical comparisons for the core experiments (vs baseline) ---


def compute_core_comparisons(core_results: dict, stats_analyzer: StatisticsAnalyzer) -> dict:
    baseline = core_results["baseline"]
    comparisons = {}
    for key, result in core_results.items():
        if key == "baseline":
            continue
        comparisons[key] = stats_analyzer.compare_experiments(
            baseline, result, metric="average_reward"
        )
    return comparisons


# --- 5. Independently replay each ablation's candidate arm ---
# `AblationRunner.run()` returns only the final `AblationResult`, not the
# intermediate `ExperimentResult`s it computed internally (and the
# framework must not be modified to expose them). To report each
# ablation's own 95% CI (not just the pairwise p-value/effect size), this
# replays each candidate arm a second time via `ExperimentRunner`, wired
# identically to how `AblationRunner` wires it internally, reusing the
# same exported policy/strategy classes
# (`RandomCriticPolicy`/`ReducedContextPolicy`/`QualityOnlyRewardStrategy`).
# No replay, resampling, or reward logic is reimplemented; this only
# repeats an existing, deterministic computation to read out an
# intermediate value the framework's public API doesn't expose.


def replay_ablation_candidate_arms(repository: InMemoryExperienceRepository) -> dict:
    def random_critic_factory(config: ExperimentConfig):
        return RandomCriticPolicy(seed=config.random_seed)

    def reduced_context_factory(config: ExperimentConfig):
        return ReducedContextPolicy(LinUCBPolicy(alpha=config.alpha or 1.0), keep_fraction=0.5)

    results = {}

    random_critic_runner = ExperimentRunner(repository=repository, policy_factory=random_critic_factory)
    results["random_critic"] = random_critic_runner.run(
        ExperimentConfig(
            experiment_name="7-candidate-random-critic",
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
            experiment_name="8-candidate-reduced-context",
            policy_name="LinUCBPolicy",
            alpha=1.0,
            random_seed=CAMPAIGN_SEED,
            num_runs=NUM_RUNS,
            candidate_actions=CANDIDATE_ACTIONS,
        )
    )

    quality_only_runner = ExperimentRunner(
        repository=repository, reward_calculator=RewardCalculator(strategy=QualityOnlyRewardStrategy())
    )
    results["quality_only_reward"] = quality_only_runner.run(
        ExperimentConfig(
            experiment_name="9-candidate-quality-only-reward",
            policy_name="LinUCBPolicy",
            alpha=1.0,
            random_seed=CAMPAIGN_SEED,
            num_runs=NUM_RUNS,
            candidate_actions=CANDIDATE_ACTIONS,
        )
    )
    return results


# --- 6. Tables ---


def _fmt_ci(ci: dict) -> str:
    return f"[{ci['lower']:.4f}, {ci['upper']:.4f}]"


def _fmt_p(p_value: float | None) -> str:
    if p_value is None:
        return "n/a"
    return f"{p_value:.4f}" if p_value >= 0.0001 else "<0.0001"


CORE_LABELS = {
    "baseline": "1. Baseline (HeuristicPolicy)",
    "linucb_alpha_0.0": "2. LinUCB alpha=0",
    "linucb_alpha_0.25": "3. LinUCB alpha=0.25",
    "linucb_alpha_0.5": "4. LinUCB alpha=0.5",
    "linucb_alpha_1.0": "5. LinUCB alpha=1.0",
    "linucb_alpha_2.0": "6. LinUCB alpha=2.0",
    "full_acrf": "10. Full ACRF (LinUCB alpha=1.0)",
}
ABLATION_LABELS = {
    "random_critic": "7. Random Critic",
    "reduced_context": "8. Reduced Context",
    "quality_only_reward": "9. Quality-only Reward",
}


def build_table1(
    core_results: dict,
    core_comparisons: dict,
    ablation_results: dict,
    ablation_candidate_results: dict,
    ablation_comparisons: dict,
) -> str:
    """Table 1: Overall comparison across all 10 experiments."""
    header = (
        "| Experiment | Avg Reward | Avg Quality | Avg Latency | Avg Iterations | "
        "Match Rate | 95% CI (reward) | vs | p-value | Effect Size | Significant |"
    )
    separator = "|---|---|---|---|---|---|---|---|---|---|---|"
    rows = [header, separator]

    for key, label in CORE_LABELS.items():
        result = core_results[key]
        ci = _fmt_ci(result.metadata["reward_confidence_interval_95"])
        if key == "baseline":
            vs, p_value, effect, sig = "-", None, "-", "-"
        else:
            comparison = core_comparisons[key]
            vs = "Baseline"
            p_value = comparison.p_value
            effect = f"{comparison.effect_size:.4f}"
            sig = comparison.significant
        rows.append(
            f"| {label} | {result.average_reward:.4f} | "
            f"{result.average_quality:.4f} | {result.average_latency:.4f} | "
            f"{result.average_iterations:.4f} | {result.match_rate:.4f} | {ci} | {vs} | "
            f"{_fmt_p(p_value)} | {effect} | {sig} |"
        )

    for key, label in ABLATION_LABELS.items():
        candidate = ablation_candidate_results[key]
        comparison = ablation_comparisons[key]
        ci = _fmt_ci(candidate.metadata["reward_confidence_interval_95"])
        rows.append(
            f"| {label} | {candidate.average_reward:.4f} | "
            f"{candidate.average_quality:.4f} | {candidate.average_latency:.4f} | "
            f"{candidate.average_iterations:.4f} | {candidate.match_rate:.4f} | {ci} | "
            f"Full ACRF | {_fmt_p(comparison.p_value)} | {comparison.effect_size:.4f} | "
            f"{comparison.significant} |"
        )
    return "\n".join(rows) + "\n"


def build_table2(ablation_results: dict, ablation_candidate_results: dict) -> str:
    """Table 2: Ablation comparison (each ablation's candidate arm vs Full ACRF)."""
    header = (
        "| Ablation | Baseline Reward (Full ACRF) | Candidate Reward | Reward Diff | "
        "Quality Diff | Latency Diff | Iteration Diff | Winner | p-value | Effect Size | "
        "Significant |"
    )
    separator = "|---|---|---|---|---|---|---|---|---|---|---|"
    rows = [header, separator]
    for key, label in ABLATION_LABELS.items():
        result = ablation_results[key]
        rows.append(
            f"| {label} | {result.baseline_reward:.4f} | {result.candidate_reward:.4f} | "
            f"{result.reward_difference:+.4f} | {result.quality_difference:+.4f} | "
            f"{result.latency_difference:+.4f} | {result.iteration_difference:+.4f} | "
            f"{result.metadata['winner']} | {_fmt_p(result.metadata['p_value'])} | "
            f"{result.metadata['effect_size']:.4f} | {result.metadata['significant']} |"
        )
    return "\n".join(rows) + "\n"


def build_table3(core_results: dict, core_comparisons: dict) -> str:
    """Table 3: Alpha sensitivity (the 5 LinUCB alpha values)."""
    header = (
        "| Alpha | Avg Reward | 95% CI (reward) | Avg Quality | Avg Latency | "
        "Avg Iterations | Match Rate | p-value vs Baseline | Effect Size |"
    )
    separator = "|---|---|---|---|---|---|---|---|---|"
    rows = [header, separator]
    for alpha in ALPHAS:
        key = f"linucb_alpha_{alpha}"
        result = core_results[key]
        comparison = core_comparisons[key]
        ci = _fmt_ci(result.metadata["reward_confidence_interval_95"])
        rows.append(
            f"| {alpha} | {result.average_reward:.4f} | {ci} | {result.average_quality:.4f} | "
            f"{result.average_latency:.4f} | {result.average_iterations:.4f} | "
            f"{result.match_rate:.4f} | {_fmt_p(comparison.p_value)} | "
            f"{comparison.effect_size:.4f} |"
        )
    return "\n".join(rows) + "\n"


# --- 7. Figures ---


def generate_figures(
    core_results: dict,
    ablation_results: dict,
    ablation_candidate_results: dict,
    figures_dir: Path,
) -> None:
    plt.rcParams["figure.autolayout"] = True

    # 1. Reward curve: per-run average_reward across the 30 bootstrap runs.
    fig, ax = plt.subplots(figsize=(8, 5))
    series = {
        "Baseline (HeuristicPolicy)": core_results["baseline"],
        "Full ACRF (LinUCB alpha=1.0)": core_results["full_acrf"],
        "Random Critic": ablation_candidate_results["random_critic"],
        "Quality-only Reward": ablation_candidate_results["quality_only_reward"],
    }
    for label, result in series.items():
        ax.plot(range(1, len(result.runs) + 1), [r.average_reward for r in result.runs], marker="o", label=label)
    ax.set_xlabel("Run index (bootstrap resample)")
    ax.set_ylabel("Average reward")
    ax.set_title("Reward curve across bootstrap runs")
    ax.legend(fontsize=8)
    fig.savefig(figures_dir / "reward_curve.png", dpi=150)
    plt.close(fig)

    # 2. Cumulative reward across the same runs.
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, result in series.items():
        values = [r.average_reward for r in result.runs]
        cumulative = [sum(values[: i + 1]) for i in range(len(values))]
        ax.plot(range(1, len(cumulative) + 1), cumulative, marker="o", label=label)
    ax.set_xlabel("Run index (bootstrap resample)")
    ax.set_ylabel("Cumulative reward")
    ax.set_title("Cumulative reward across bootstrap runs")
    ax.legend(fontsize=8)
    fig.savefig(figures_dir / "cumulative_reward.png", dpi=150)
    plt.close(fig)

    # 3. Reward vs Alpha.
    fig, ax = plt.subplots(figsize=(7, 5))
    rewards = [core_results[f"linucb_alpha_{a}"].average_reward for a in ALPHAS]
    lowers = [
        core_results[f"linucb_alpha_{a}"].metadata["reward_confidence_interval_95"]["lower"]
        for a in ALPHAS
    ]
    uppers = [
        core_results[f"linucb_alpha_{a}"].metadata["reward_confidence_interval_95"]["upper"]
        for a in ALPHAS
    ]
    lower_err = [r - lo for r, lo in zip(rewards, lowers, strict=True)]
    upper_err = [hi - r for r, hi in zip(rewards, uppers, strict=True)]
    ax.errorbar(ALPHAS, rewards, yerr=[lower_err, upper_err], marker="o", capsize=4)
    ax.axhline(
        core_results["baseline"].average_reward, color="gray", linestyle="--", label="Baseline (HeuristicPolicy)"
    )
    ax.set_xlabel("LinUCB alpha")
    ax.set_ylabel("Average reward (95% CI)")
    ax.set_title("Reward vs. alpha")
    ax.legend(fontsize=8)
    fig.savefig(figures_dir / "reward_vs_alpha.png", dpi=150)
    plt.close(fig)

    # 4. Critic selection frequency.
    fig, ax = plt.subplots(figsize=(8, 5))
    freq_series = {
        "Baseline": core_results["baseline"].critic_selection_frequency,
        "Full ACRF": core_results["full_acrf"].critic_selection_frequency,
        "Random Critic": ablation_candidate_results["random_critic"].critic_selection_frequency,
    }
    critics = CANDIDATE_ACTIONS
    width = 0.25
    x = range(len(critics))
    for offset, (label, freq) in zip((-1, 0, 1), freq_series.items(), strict=True):
        heights = [freq.get(critic, 0.0) for critic in critics]
        ax.bar([xi + offset * width for xi in x], heights, width=width, label=label)
    ax.set_xticks(list(x))
    ax.set_xticklabels(critics)
    ax.set_ylabel("Selection frequency")
    ax.set_title("Critic selection frequency")
    ax.legend(fontsize=8)
    fig.savefig(figures_dir / "critic_selection_frequency.png", dpi=150)
    plt.close(fig)

    # 5. Latency comparison across all 10 experiments.
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = list(CORE_LABELS.values()) + list(ABLATION_LABELS.values())
    latencies = [core_results[k].average_latency for k in CORE_LABELS] + [
        ablation_candidate_results[k].average_latency for k in ABLATION_LABELS
    ]
    ax.bar(range(len(labels)), latencies, color="tab:orange")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Average latency")
    ax.set_title("Latency comparison across all experiments")
    fig.savefig(figures_dir / "latency_comparison.png", dpi=150)
    plt.close(fig)

    # 6. Iterations comparison across all 10 experiments.
    fig, ax = plt.subplots(figsize=(10, 5))
    iterations = [core_results[k].average_iterations for k in CORE_LABELS] + [
        ablation_candidate_results[k].average_iterations for k in ABLATION_LABELS
    ]
    ax.bar(range(len(labels)), iterations, color="tab:green")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Average iterations")
    ax.set_title("Iterations comparison across all experiments")
    fig.savefig(figures_dir / "iterations_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    print(f"Generating synthetic experience log (seed={DATA_SEED}, n={NUM_RECORDS})...")
    repository = generate_repository(DATA_SEED, NUM_RECORDS)
    print(f"Repository size: {repository.count()}")

    print("Running core experiments (baseline + alpha sweep + Full ACRF)...")
    core_results = run_core_experiments(repository)

    print("Running ablations (Random Critic, Reduced Context, Quality-only Reward)...")
    ablation_results = run_ablations(repository)

    print("Replaying ablation candidate arms independently (for per-arm 95% CI)...")
    ablation_candidate_results = replay_ablation_candidate_arms(repository)
    for key in ABLATION_LABELS:
        assert ablation_candidate_results[key].average_reward == ablation_results[key].candidate_reward, (
            f"independent replay of {key!r} diverged from AblationRunner's own candidate arm"
        )

    print("Computing statistical comparisons...")
    stats_analyzer = StatisticsAnalyzer()
    core_comparisons = compute_core_comparisons(core_results, stats_analyzer)
    ablation_comparisons = {
        key: stats_analyzer.compare_experiments(
            core_results["full_acrf"], ablation_candidate_results[key], metric="average_reward"
        )
        for key in ABLATION_LABELS
    }

    print(f"Repository count after campaign (must equal {NUM_RECORDS}): {repository.count()}")
    assert repository.count() == NUM_RECORDS, "source repository was mutated during the campaign"

    print("Building tables...")
    table1 = build_table1(
        core_results, core_comparisons, ablation_results, ablation_candidate_results, ablation_comparisons
    )
    table2 = build_table2(ablation_results, ablation_candidate_results)
    table3 = build_table3(core_results, core_comparisons)
    (RESULTS_DIR / "table1_overall_comparison.md").write_text(table1, encoding="utf-8")
    (RESULTS_DIR / "table2_ablation_comparison.md").write_text(table2, encoding="utf-8")
    (RESULTS_DIR / "table3_alpha_sensitivity.md").write_text(table3, encoding="utf-8")

    print("Generating figures...")
    generate_figures(core_results, ablation_results, ablation_candidate_results, FIGURES_DIR)

    print("Exporting raw results (JSON/CSV/Markdown) via existing Exporter/AblationReportGenerator...")
    experiment_exporter = ExperimentExporter()
    all_experiment_results = list(core_results.values())
    (RESULTS_DIR / "core_experiments.json").write_text(
        experiment_exporter.to_json(all_experiment_results), encoding="utf-8"
    )
    (RESULTS_DIR / "core_experiments.csv").write_text(
        experiment_exporter.to_csv(all_experiment_results), encoding="utf-8"
    )
    (RESULTS_DIR / "core_experiments_summary.md").write_text(
        experiment_exporter.to_markdown(all_experiment_results), encoding="utf-8"
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

    print("Campaign complete. Results written to:", RESULTS_DIR)
    print("Figures written to:", FIGURES_DIR)
