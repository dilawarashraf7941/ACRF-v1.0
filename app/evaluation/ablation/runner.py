"""`AblationRunner`: automates ablation studies over the existing evaluation pipeline.

Every ablation "arm" (a baseline or a candidate) is run by constructing
an `app.evaluation.experiments.ExperimentConfig` and replaying it through
an `app.evaluation.experiments.ExperimentRunner` — which itself replays
via `app.evaluation.offline.ReplayEngine` and aggregates via
`app.evaluation.offline.OfflineEvaluator`. The two resulting
`ExperimentResult`s are then compared via
`app.evaluation.offline.Benchmark` (for the four delta fields) and
`app.evaluation.statistics.Analyzer` (for statistical significance).
**No replay, resampling, reward, or statistical-test logic is
implemented in this module** — every one of those pieces is reused,
unmodified, from the frameworks built in prior work.

Two of the seven supported ablation types (`random_critic_selection`,
`reduced_context_features`) need a policy this framework didn't already
ship; one (`alternative_reward_definitions`) needs a reward *definition*
this framework didn't already ship. Both `ExperimentRunner` (via its
`policy_factory` constructor argument) and `RewardCalculator` (via its
`strategy` constructor argument) were **already** built with exactly
this kind of extension in mind — see their own docstrings. This module
supplies three small, self-contained implementations of those existing
extension points (`RandomCriticPolicy`, `ReducedContextPolicy`,
`QualityOnlyRewardStrategy`) and nothing else. No graph integration, no
live/online learning, no PPO, and no reinforcement learning.
"""

import math
import random
from collections.abc import Callable

from app.context import ContextVector
from app.evaluation.ablation.models import AblationConfig, AblationResult
from app.evaluation.experiments import ExperimentConfig, ExperimentResult, ExperimentRunner
from app.evaluation.experiments.runner import _default_policy_factory
from app.evaluation.offline import Benchmark, ReplayablePolicy, ReplayResult
from app.evaluation.statistics import Analyzer as StatisticsAnalyzer
from app.experience import ExperienceRecord, ExperienceRepository
from app.policy.models import PolicyDecision
from app.reward import RewardCalculator
from app.reward.models import RewardSignal
from app.reward.strategy import BaseRewardStrategy

DEFAULT_ALPHA_SWEEP: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
"""The example alpha values named by the task spec, used by `AblationRunner.standard_configs`."""

DEFAULT_KEEP_FEATURE_FRACTION = 0.5
"""The default fraction of context features `reduced_context_features` keeps."""

_SUPPORTED_ABLATION_TYPES: tuple[str, ...] = (
    "no_exploration",
    "alpha_sweep",
    "random_critic_selection",
    "heuristic_only",
    "linucb_only",
    "reduced_context_features",
    "alternative_reward_definitions",
)


# --- new policies, realized entirely through the existing ReplayablePolicy extension point ---


class RandomCriticPolicy:
    """Selects a critic uniformly at random from the candidate set — a naive baseline.

    Deterministic given a fixed seed: holds its own `random.Random`,
    seeded once at construction and consumed sequentially across
    `select_action` calls, so replaying the same experiences in the same
    order always produces the same selections. Returns
    `app.policy.models.PolicyDecision` — an existing, unmodified model —
    rather than inventing a new decision type.
    """

    policy_name = "RandomCriticPolicy"
    policy_version = "1.0.0"

    def __init__(self, seed: int) -> None:
        """Create a policy whose random choices are fully determined by `seed`."""
        self._rng = random.Random(seed)

    def select_action(self, context: ContextVector, candidate_actions: list[str]) -> PolicyDecision:
        """Select one candidate uniformly at random.

        Args:
            context: Unused for selection (recorded only in `metadata`)
                — random selection is, by definition, context-independent.
            candidate_actions: The critics to choose from.

        Returns:
            A `PolicyDecision` naming the chosen critic, with a uniform
            score assigned to it and `0.0` to every other candidate.
        """
        if not candidate_actions:
            return PolicyDecision(
                selected_critics=[],
                scores={},
                ranking=[],
                policy_name=self.policy_name,
                policy_version=self.policy_version,
                confidence=0.0,
                metadata={"context_id": context.context_id, "selection_strategy": "uniform_random"},
            )

        chosen = self._rng.choice(candidate_actions)
        scores = {name: (1.0 if name == chosen else 0.0) for name in candidate_actions}
        others = [name for name in candidate_actions if name != chosen]
        ranking = [{"critic_name": chosen, "score": 1.0, "rank": 1}] + [
            {"critic_name": name, "score": 0.0, "rank": index + 2}
            for index, name in enumerate(others)
        ]
        return PolicyDecision(
            selected_critics=[chosen],
            scores=scores,
            ranking=ranking,
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            confidence=1.0 / len(candidate_actions),
            metadata={"context_id": context.context_id, "selection_strategy": "uniform_random"},
        )


class ReducedContextPolicy:
    """Wraps an existing `ReplayablePolicy`, masking most context features before delegating.

    Realizes the "Reduced Context Features" ablation without modifying
    `ContextVector`, `ReplayEngine`, or any policy class: keeps only the
    first `keep_fraction` (by count, rounded up, at least one) of
    `context.feature_order`, in the same order, and builds a **new**
    `ContextVector` (via `model_copy` — the original is never mutated)
    containing only those features before delegating to `wrapped`.
    """

    def __init__(
        self, wrapped: ReplayablePolicy, keep_fraction: float = DEFAULT_KEEP_FEATURE_FRACTION
    ) -> None:
        """Create a wrapper around `wrapped` that masks context features.

        Args:
            wrapped: The policy to delegate to, after feature reduction.
            keep_fraction: The fraction of `feature_order` to keep,
                counted from the front. Must be in `(0.0, 1.0]`.

        Raises:
            ValueError: If `keep_fraction` is outside `(0.0, 1.0]`.
        """
        if not 0.0 < keep_fraction <= 1.0:
            raise ValueError(f"keep_fraction must be in (0.0, 1.0], got {keep_fraction}")
        self._wrapped = wrapped
        self._keep_fraction = keep_fraction

    def select_action(self, context: ContextVector, candidate_actions: list[str]) -> object:
        """Reduce `context`'s features, then delegate to the wrapped policy."""
        return self._wrapped.select_action(self._reduce(context), candidate_actions)

    def _reduce(self, context: ContextVector) -> ContextVector:
        keep_count = max(1, math.ceil(len(context.feature_order) * self._keep_fraction))
        kept_order = context.feature_order[:keep_count]
        kept_features = {name: context.features[name] for name in kept_order}
        return context.model_copy(update={"features": kept_features, "feature_order": kept_order})


# --- a new reward definition, via the existing BaseRewardStrategy extension point ---


class QualityOnlyRewardStrategy(BaseRewardStrategy):
    """Reward = `aggregated_quality_score` alone; cost/latency/completion/correction ignored.

    A genuinely different reward *definition* for the "Alternative Reward
    Definitions" ablation, realized entirely through the existing,
    sanctioned `BaseRewardStrategy` extension point (see
    `app/reward/strategy.py`'s own docstring: "a future caller ... can
    supply a different BaseRewardStrategy without any change to this
    class") — `app/reward` is not modified.
    """

    strategy_name = "QualityOnlyRewardStrategy"

    def compute(self, experience: ExperienceRecord) -> RewardSignal:
        """Compute a `RewardSignal` from `experience.aggregated_quality_score` alone."""
        quality = experience.aggregated_quality_score
        quality_reward = max(0.0, min(1.0, quality)) if quality is not None else 0.0
        return RewardSignal(
            reward=round(quality_reward, 6),
            quality_reward=round(quality_reward, 6),
            efficiency_penalty=0.0,
            cost_penalty=0.0,
            latency_penalty=0.0,
            correction_penalty=0.0,
            completion_bonus=0.0,
            confidence=1.0 if quality is not None else 0.0,
            strategy=self.strategy_name,
            explanation=(
                f"reward={quality_reward:.4f} = quality_reward only "
                "(cost/latency/completion/correction penalties ignored by design)."
            ),
            metadata={"experience_id": experience.experience_id},
        )


# --- adapters: reshape ExperimentResult into the shape Benchmark already knows how to compare ---


def _experiment_result_to_replay_result(result: ExperimentResult, label: str) -> ReplayResult:
    """Build a `ReplayResult` summarizing `result`, for feeding into `Benchmark.compare`.

    Pure data reshaping — every number is copied directly from `result`;
    none of `Benchmark`'s comparison arithmetic is reimplemented here.
    `label` (not `result.policy_name`) becomes `ReplayResult.policy_name`,
    so `BenchmarkResult.winner` is always unambiguous even when the
    baseline and candidate arms share the same underlying policy name
    (e.g. `alpha_sweep`, `reduced_context_features`).
    """
    num_runs = len(result.runs)
    return ReplayResult(
        policy_name=label,
        total_experiences=sum(run.total_experiences for run in result.runs),
        total_reward=result.average_reward * num_runs,
        average_reward=result.average_reward,
        average_quality=result.average_quality,
        average_iterations=result.average_iterations,
        average_latency=result.average_latency,
        critic_selection_frequency=result.critic_selection_frequency,
        metadata={
            "source": "ExperimentResult",
            "experiment_name": result.experiment_name,
            "policy_name": result.policy_name,
            "num_runs": num_runs,
        },
    )


def _build_conclusion(
    config: AblationConfig, winner_label: str, significant: bool, p_value: float, effect_size: float
) -> str:
    """Format a plain-language conclusion from already-computed Benchmark/Analyzer outputs.

    Pure string templating over numbers computed elsewhere — no
    statistical or comparison logic lives here. When `baseline_policy`
    and `candidate_policy` share the same name (e.g. `alpha_sweep`,
    `alternative_reward_definitions` — the same underlying policy, only
    a hyperparameter or reward definition differs), the winner is
    additionally tagged `(baseline)`/`(candidate)` so the two arms stay
    distinguishable in the sentence.
    """
    same_name = config.baseline_policy == config.candidate_policy
    baseline_label = (
        f"{config.baseline_policy} (baseline)" if same_name else config.baseline_policy
    )
    candidate_label = (
        f"{config.candidate_policy} (candidate)" if same_name else config.candidate_policy
    )

    winner_name = {
        "baseline": baseline_label,
        "candidate": candidate_label,
        "tie": "neither arm",
    }.get(winner_label, winner_label)

    if not significant:
        return (
            f"No statistically significant difference between {baseline_label} "
            f"and {candidate_label} for the '{config.ablation_type}' ablation "
            f"(p={p_value:.4f})."
        )
    return (
        f"{winner_name} performed significantly better in the '{config.ablation_type}' "
        f"ablation (p={p_value:.4f}, effect size={effect_size:.4f})."
    )


class AblationRunner:
    """Runs one or more ablation studies by composing existing evaluation components.

    Constructed via dependency injection with the `ExperienceRepository`
    every arm reads from, plus optional overrides for every collaborator.
    `AblationRunner` itself never touches replay, resampling, reward, or
    statistical-test math directly — it only builds `ExperimentConfig`s,
    picks which (already-existing) `ExperimentRunner` wiring each
    ablation type needs, and reads the fields off the `BenchmarkResult`/
    `StatisticalComparison` those existing components already compute.
    """

    def __init__(
        self,
        repository: ExperienceRepository,
        reward_calculator: RewardCalculator | None = None,
        benchmark: Benchmark | None = None,
        statistics_analyzer: StatisticsAnalyzer | None = None,
    ) -> None:
        """Create a runner wired to `repository` and its collaborators.

        Args:
            repository: The source `ExperienceRepository` every arm reads
                from. Never written to by this class or by anything it
                constructs.
            reward_calculator: The *baseline* reward calculator (using
                the default `WeightedRewardStrategy` unless overridden).
                The `alternative_reward_definitions` ablation's candidate
                arm always uses `QualityOnlyRewardStrategy` regardless of
                this argument — that comparison is the entire point of
                that ablation type.
            benchmark: Computes the four delta fields. Defaults to a
                plain `Benchmark()`.
            statistics_analyzer: Computes significance/effect size.
                Defaults to a plain `app.evaluation.statistics.Analyzer()`.
        """
        self._repository = repository
        self._reward_calculator = (
            reward_calculator if reward_calculator is not None else RewardCalculator()
        )
        self._benchmark = benchmark if benchmark is not None else Benchmark()
        self._statistics_analyzer = (
            statistics_analyzer if statistics_analyzer is not None else StatisticsAnalyzer()
        )

        self._standard_runner = ExperimentRunner(
            repository=self._repository, reward_calculator=self._reward_calculator
        )
        self._random_critic_runner = ExperimentRunner(
            repository=self._repository,
            reward_calculator=self._reward_calculator,
            policy_factory=self._random_critic_policy_factory,
        )
        self._reduced_context_runner = ExperimentRunner(
            repository=self._repository,
            reward_calculator=self._reward_calculator,
            policy_factory=self._reduced_context_policy_factory,
        )
        self._alternative_reward_runner = ExperimentRunner(
            repository=self._repository,
            reward_calculator=RewardCalculator(strategy=QualityOnlyRewardStrategy()),
        )

        self._arm_builders: dict[
            str, Callable[..., tuple[ExperimentResult, ExperimentResult]]
        ] = {
            "no_exploration": self._run_no_exploration,
            "alpha_sweep": self._run_alpha_sweep_arm,
            "random_critic_selection": self._run_random_critic_selection,
            "heuristic_only": self._run_heuristic_only,
            "linucb_only": self._run_linucb_only,
            "reduced_context_features": self._run_reduced_context_features,
            "alternative_reward_definitions": self._run_alternative_reward_definitions,
        }

    # --- custom policy factories (the sanctioned ExperimentRunner extension point) ---

    def _random_critic_policy_factory(self, config: ExperimentConfig) -> ReplayablePolicy:
        if config.policy_name == "RandomCriticPolicy":
            return RandomCriticPolicy(seed=config.random_seed)
        return _default_policy_factory(config)

    def _reduced_context_policy_factory(self, config: ExperimentConfig) -> ReplayablePolicy:
        keep_fraction = float(
            config.metadata.get("keep_feature_fraction", DEFAULT_KEEP_FEATURE_FRACTION)
        )
        wrapped = _default_policy_factory(config)
        return ReducedContextPolicy(wrapped, keep_fraction=keep_fraction)

    # --- public API ---

    def run(
        self,
        config: AblationConfig,
        *,
        num_runs: int,
        random_seed: int,
        candidate_actions: list[str] | None = None,
    ) -> AblationResult:
        """Run one ablation comparison end to end.

        Args:
            config: Which ablation to run and how (see `models.py`).
            num_runs: Passed through to both arms' `ExperimentConfig.num_runs`.
            random_seed: Passed through to both arms' `ExperimentConfig.random_seed`
                (so both arms' bootstrap resamples are identical, and the
                comparison is genuinely paired — see
                `app/evaluation/statistics/README.md`).
            candidate_actions: Passed through to both arms'
                `ExperimentConfig.candidate_actions`.

        Returns:
            The resulting `AblationResult`.

        Raises:
            ValueError: If `config.ablation_type` is not one of the
                seven supported types.
        """
        builder = self._arm_builders.get(config.ablation_type)
        if builder is None:
            raise ValueError(
                f"Unsupported ablation_type {config.ablation_type!r}. "
                f"Supported: {sorted(self._arm_builders)}."
            )

        baseline_result, candidate_result = builder(
            config, num_runs=num_runs, random_seed=random_seed, candidate_actions=candidate_actions
        )

        baseline_replay = _experiment_result_to_replay_result(baseline_result, "baseline")
        candidate_replay = _experiment_result_to_replay_result(candidate_result, "candidate")
        benchmark_result = self._benchmark.compare(
            baseline=baseline_replay, candidate=candidate_replay
        )

        statistical_comparison = self._statistics_analyzer.compare_experiments(
            baseline_result, candidate_result, metric="average_reward"
        )

        conclusion = _build_conclusion(
            config,
            benchmark_result.winner,
            statistical_comparison.significant,
            statistical_comparison.p_value,
            statistical_comparison.effect_size,
        )
        winner_label = {
            "baseline": config.baseline_policy,
            "candidate": config.candidate_policy,
            "tie": "tie",
        }.get(benchmark_result.winner, benchmark_result.winner)

        return AblationResult(
            ablation_type=config.ablation_type,
            baseline_reward=baseline_result.average_reward,
            candidate_reward=candidate_result.average_reward,
            reward_difference=benchmark_result.reward_improvement,
            quality_difference=benchmark_result.quality_improvement,
            latency_difference=benchmark_result.latency_difference,
            iteration_difference=benchmark_result.iteration_difference,
            conclusion=conclusion,
            metadata={
                "experiment_name": config.experiment_name,
                "baseline_policy": config.baseline_policy,
                "candidate_policy": config.candidate_policy,
                "winner": winner_label,
                "p_value": statistical_comparison.p_value,
                "effect_size": statistical_comparison.effect_size,
                "test_used": statistical_comparison.test_used,
                "significant": statistical_comparison.significant,
                "sample_size": statistical_comparison.sample_size,
                "num_runs": num_runs,
                "random_seed": random_seed,
                **config.metadata,
            },
        )

    def run_all(
        self,
        experiment_name: str,
        *,
        num_runs: int,
        random_seed: int,
        candidate_actions: list[str] | None = None,
        alphas: tuple[float, ...] = DEFAULT_ALPHA_SWEEP,
    ) -> list[AblationResult]:
        """Run every standard ablation config (`standard_configs`) and return the results.

        Args:
            experiment_name: Prefix used to label every generated config.
            num_runs: Passed through to every arm.
            random_seed: Passed through to every arm.
            candidate_actions: Passed through to every arm.
            alphas: The alpha values `alpha_sweep` covers.

        Returns:
            One `AblationResult` per config from `standard_configs`, in order.
        """
        configs = self.standard_configs(experiment_name, alphas=alphas)
        return [
            self.run(
                config,
                num_runs=num_runs,
                random_seed=random_seed,
                candidate_actions=candidate_actions,
            )
            for config in configs
        ]

    @staticmethod
    def standard_configs(
        experiment_name: str, alphas: tuple[float, ...] = DEFAULT_ALPHA_SWEEP
    ) -> list[AblationConfig]:
        """Build one `AblationConfig` per supported ablation type, with sensible defaults.

        Args:
            experiment_name: Prefix used to label every generated config.
            alphas: The alpha values `alpha_sweep` covers (one config per value).

        Returns:
            `1 + len(alphas) + 5` configs: `no_exploration`, one
            `alpha_sweep` config per value in `alphas`,
            `random_critic_selection`, `heuristic_only`, `linucb_only`,
            `reduced_context_features`, and `alternative_reward_definitions`.
        """
        configs = [
            AblationConfig(
                experiment_name=f"{experiment_name}-no-exploration",
                baseline_policy="LinUCBPolicy",
                candidate_policy="LinUCBPolicy",
                ablation_type="no_exploration",
            )
        ]
        configs.extend(
            AblationConfig(
                experiment_name=f"{experiment_name}-alpha-{alpha}",
                baseline_policy="LinUCBPolicy",
                candidate_policy="LinUCBPolicy",
                ablation_type="alpha_sweep",
                metadata={"alpha": alpha},
            )
            for alpha in alphas
        )
        configs.append(
            AblationConfig(
                experiment_name=f"{experiment_name}-random-critic",
                baseline_policy="HeuristicPolicy",
                candidate_policy="RandomCriticPolicy",
                ablation_type="random_critic_selection",
            )
        )
        configs.append(
            AblationConfig(
                experiment_name=f"{experiment_name}-heuristic-only",
                baseline_policy="LinUCBPolicy",
                candidate_policy="HeuristicPolicy",
                ablation_type="heuristic_only",
            )
        )
        configs.append(
            AblationConfig(
                experiment_name=f"{experiment_name}-linucb-only",
                baseline_policy="HeuristicPolicy",
                candidate_policy="LinUCBPolicy",
                ablation_type="linucb_only",
            )
        )
        configs.append(
            AblationConfig(
                experiment_name=f"{experiment_name}-reduced-context",
                baseline_policy="LinUCBPolicy",
                candidate_policy="LinUCBPolicy",
                ablation_type="reduced_context_features",
            )
        )
        configs.append(
            AblationConfig(
                experiment_name=f"{experiment_name}-alternative-reward",
                baseline_policy="LinUCBPolicy",
                candidate_policy="LinUCBPolicy",
                ablation_type="alternative_reward_definitions",
            )
        )
        return configs

    # --- per-ablation-type arm builders ---

    @staticmethod
    def _make_experiment_config(
        experiment_name: str,
        policy_name: str,
        alpha: float | None,
        num_runs: int,
        random_seed: int,
        candidate_actions: list[str] | None,
        metadata: dict[str, object] | None = None,
    ) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_name=experiment_name,
            policy_name=policy_name,
            alpha=alpha,
            random_seed=random_seed,
            num_runs=num_runs,
            candidate_actions=candidate_actions,
            metadata=metadata or {},
        )

    def _run_no_exploration(
        self, config: AblationConfig, *, num_runs, random_seed, candidate_actions
    ) -> tuple[ExperimentResult, ExperimentResult]:
        baseline_alpha = float(config.metadata.get("baseline_alpha", 1.0))
        baseline_cfg = self._make_experiment_config(
            f"{config.experiment_name}-baseline",
            config.baseline_policy,
            baseline_alpha,
            num_runs,
            random_seed,
            candidate_actions,
        )
        candidate_cfg = self._make_experiment_config(
            f"{config.experiment_name}-candidate",
            config.candidate_policy,
            0.0,
            num_runs,
            random_seed,
            candidate_actions,
        )
        return self._standard_runner.run(baseline_cfg), self._standard_runner.run(candidate_cfg)

    def _run_alpha_sweep_arm(
        self, config: AblationConfig, *, num_runs, random_seed, candidate_actions
    ) -> tuple[ExperimentResult, ExperimentResult]:
        if "alpha" not in config.metadata:
            raise ValueError("The 'alpha_sweep' ablation requires metadata['alpha'].")
        baseline_alpha = float(config.metadata.get("baseline_alpha", 1.0))
        candidate_alpha = float(config.metadata["alpha"])
        baseline_cfg = self._make_experiment_config(
            f"{config.experiment_name}-baseline",
            config.baseline_policy,
            baseline_alpha,
            num_runs,
            random_seed,
            candidate_actions,
        )
        candidate_cfg = self._make_experiment_config(
            f"{config.experiment_name}-candidate",
            config.candidate_policy,
            candidate_alpha,
            num_runs,
            random_seed,
            candidate_actions,
        )
        return self._standard_runner.run(baseline_cfg), self._standard_runner.run(candidate_cfg)

    def _run_random_critic_selection(
        self, config: AblationConfig, *, num_runs, random_seed, candidate_actions
    ) -> tuple[ExperimentResult, ExperimentResult]:
        baseline_alpha = (
            float(config.metadata["baseline_alpha"])
            if "baseline_alpha" in config.metadata
            else None
        )
        baseline_cfg = self._make_experiment_config(
            f"{config.experiment_name}-baseline",
            config.baseline_policy,
            baseline_alpha,
            num_runs,
            random_seed,
            candidate_actions,
        )
        candidate_cfg = self._make_experiment_config(
            f"{config.experiment_name}-candidate",
            "RandomCriticPolicy",
            None,
            num_runs,
            random_seed,
            candidate_actions,
        )
        return (
            self._standard_runner.run(baseline_cfg),
            self._random_critic_runner.run(candidate_cfg),
        )

    def _run_heuristic_only(
        self, config: AblationConfig, *, num_runs, random_seed, candidate_actions
    ) -> tuple[ExperimentResult, ExperimentResult]:
        baseline_alpha = float(config.metadata.get("baseline_alpha", 1.0))
        baseline_cfg = self._make_experiment_config(
            f"{config.experiment_name}-baseline",
            config.baseline_policy,
            baseline_alpha,
            num_runs,
            random_seed,
            candidate_actions,
        )
        candidate_cfg = self._make_experiment_config(
            f"{config.experiment_name}-candidate",
            config.candidate_policy,
            None,
            num_runs,
            random_seed,
            candidate_actions,
        )
        return self._standard_runner.run(baseline_cfg), self._standard_runner.run(candidate_cfg)

    def _run_linucb_only(
        self, config: AblationConfig, *, num_runs, random_seed, candidate_actions
    ) -> tuple[ExperimentResult, ExperimentResult]:
        candidate_alpha = float(config.metadata.get("candidate_alpha", 1.0))
        baseline_cfg = self._make_experiment_config(
            f"{config.experiment_name}-baseline",
            config.baseline_policy,
            None,
            num_runs,
            random_seed,
            candidate_actions,
        )
        candidate_cfg = self._make_experiment_config(
            f"{config.experiment_name}-candidate",
            config.candidate_policy,
            candidate_alpha,
            num_runs,
            random_seed,
            candidate_actions,
        )
        return self._standard_runner.run(baseline_cfg), self._standard_runner.run(candidate_cfg)

    def _run_reduced_context_features(
        self, config: AblationConfig, *, num_runs, random_seed, candidate_actions
    ) -> tuple[ExperimentResult, ExperimentResult]:
        alpha = float(config.metadata.get("alpha", 1.0))
        keep_fraction = float(
            config.metadata.get("keep_feature_fraction", DEFAULT_KEEP_FEATURE_FRACTION)
        )
        baseline_cfg = self._make_experiment_config(
            f"{config.experiment_name}-baseline",
            config.baseline_policy,
            alpha,
            num_runs,
            random_seed,
            candidate_actions,
        )
        candidate_cfg = self._make_experiment_config(
            f"{config.experiment_name}-candidate",
            config.candidate_policy,
            alpha,
            num_runs,
            random_seed,
            candidate_actions,
            metadata={"keep_feature_fraction": keep_fraction},
        )
        return (
            self._standard_runner.run(baseline_cfg),
            self._reduced_context_runner.run(candidate_cfg),
        )

    def _run_alternative_reward_definitions(
        self, config: AblationConfig, *, num_runs, random_seed, candidate_actions
    ) -> tuple[ExperimentResult, ExperimentResult]:
        alpha = (
            float(config.metadata.get("alpha", 1.0))
            if config.baseline_policy == "LinUCBPolicy"
            else None
        )
        baseline_cfg = self._make_experiment_config(
            f"{config.experiment_name}-baseline",
            config.baseline_policy,
            alpha,
            num_runs,
            random_seed,
            candidate_actions,
        )
        candidate_cfg = self._make_experiment_config(
            f"{config.experiment_name}-candidate",
            config.candidate_policy,
            alpha,
            num_runs,
            random_seed,
            candidate_actions,
        )
        return (
            self._standard_runner.run(baseline_cfg),
            self._alternative_reward_runner.run(candidate_cfg),
        )
