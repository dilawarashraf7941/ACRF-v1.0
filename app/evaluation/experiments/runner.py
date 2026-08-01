"""`ExperimentRunner`: automates reproducible experiments over the existing
Offline Replay Framework (`app/evaluation/offline`).

`ReplayEngine.replay()` is a pure, deterministic function of a
repository's contents and a policy's current (never-updated-during-
replay) state — see `app/evaluation/offline/replay.py`. Replaying the
same policy against the same repository twice always produces the exact
same `ReplayResult`. That determinism is exactly what makes offline
replay trustworthy, but it also means "run the replay framework multiple
times" cannot produce `N` genuinely *independent* runs by simply calling
`ReplayEngine.replay()` in a loop — every call would return an identical
result, making `std_reward`, `minimum`/`maximum`, and a 95% confidence
interval all degenerate to a single point.

For `num_runs > 1`, `ExperimentRunner` instead draws `num_runs`
independent **bootstrap resamples** (with replacement, seeded by
`ExperimentConfig.random_seed`) of the source repository's stored
experiences, and replays each resample separately. This is a standard,
well-established technique for estimating the variance/confidence
interval of an off-policy value estimate from a fixed batch of logged
data — it requires no change to `ReplayEngine`, `ExperienceRepository`,
or any policy, and it never mutates the source repository or any
`ExperienceRecord` (`ExperienceRecord` is frozen; a resample only holds
new references to the same immutable records). For `num_runs == 1`, no
resampling happens at all: the source data is replayed directly, exactly
once.

No graph integration, no live/online learning, no policy updates during
an experiment (this module never calls `.update()` on any policy — the
Research Constraint "do not update policy during experiments unless
explicitly configured" is satisfied by there being no update code path
at all, not by a flag that defaults to off), no PPO, and no
reinforcement learning are implemented anywhere in this module.
"""

import random
from collections.abc import Callable

from app.evaluation.experiments.analyzer import Analyzer
from app.evaluation.experiments.models import ExperimentConfig, ExperimentResult
from app.evaluation.offline import OfflineEvaluator, ReplayablePolicy, ReplayEngine, ReplayResult
from app.experience import ExperienceRecord, ExperienceRepository
from app.policy import HeuristicPolicy
from app.policy.linucb import LinUCBPolicy
from app.reward import RewardCalculator


class BootstrapExperienceRepository(ExperienceRepository):
    """An in-memory `ExperienceRepository` over an explicit list of records.

    Unlike `InMemoryExperienceRepository` (see `app/experience/repository.py`),
    this implementation does **not** require `experience_id` uniqueness —
    `add` simply appends. That is exactly what representing a bootstrap
    resample (the same `ExperienceRecord` legitimately drawn more than
    once) requires; `InMemoryExperienceRepository.add` would raise on the
    second occurrence of a repeated id. No `ExperienceRecord` is ever
    copied or mutated here — this class only holds references to
    whichever (already-frozen) records it is constructed or `add`-ed with.

    This is a new implementation of the existing, sanctioned
    `ExperienceRepository` abstract interface (see its own docstring:
    "a future ... implementation can be substituted without requiring
    any change"), not a modification of `app/experience`.
    """

    def __init__(self, records: list[ExperienceRecord] | None = None) -> None:
        """Create a repository pre-populated with `records` (or empty)."""
        self._records: list[ExperienceRecord] = list(records) if records is not None else []

    def add(self, record: ExperienceRecord) -> None:
        """Append `record`, regardless of whether its `experience_id` is already present."""
        self._records.append(record)

    def get(self, experience_id: str) -> ExperienceRecord | None:
        """Return the first stored record with a matching `experience_id`, or `None`."""
        for record in self._records:
            if record.experience_id == experience_id:
                return record
        return None

    def list(self) -> list[ExperienceRecord]:
        """Return every stored record, in insertion order (duplicates included)."""
        return list(self._records)

    def clear(self) -> None:
        """Remove every stored record."""
        self._records.clear()

    def count(self) -> int:
        """Return the number of stored records (duplicates counted separately)."""
        return len(self._records)


def _default_policy_factory(config: ExperimentConfig) -> ReplayablePolicy:
    """Build a fresh policy instance for `config.policy_name`.

    Supports the two policies this framework is asked to support:
    `"HeuristicPolicy"` and `"LinUCBPolicy"` (constructed with
    `config.alpha`, defaulting to `1.0` when unset). A caller needing any
    other policy should inject a custom `policy_factory` into
    `ExperimentRunner` rather than expecting this function to change.

    Args:
        config: The experiment configuration naming the policy.

    Returns:
        A freshly constructed, untrained policy instance.

    Raises:
        ValueError: If `config.policy_name` is neither of the two
            supported names.
    """
    if config.policy_name == "HeuristicPolicy":
        return HeuristicPolicy()
    if config.policy_name == "LinUCBPolicy":
        alpha = config.alpha if config.alpha is not None else 1.0
        return LinUCBPolicy(alpha=alpha)
    raise ValueError(
        f"Unrecognized policy_name {config.policy_name!r}. The built-in policy factory "
        "supports 'HeuristicPolicy' and 'LinUCBPolicy'; inject a custom policy_factory "
        "into ExperimentRunner to support any other policy."
    )


class ExperimentRunner:
    """Runs one or more independent replays of a policy and aggregates the results.

    Constructed via dependency injection: an `ExperienceRepository` to
    read the source data from, plus optional overrides for every
    collaborator (`RewardCalculator`, `OfflineEvaluator`, `Analyzer`, and
    the policy-construction function itself). A fresh policy instance is
    built for every run via `policy_factory` — never reused, and never
    `.update()`-d — so no state can leak between runs or between
    experiments.
    """

    def __init__(
        self,
        repository: ExperienceRepository,
        reward_calculator: RewardCalculator | None = None,
        evaluator: OfflineEvaluator | None = None,
        analyzer: Analyzer | None = None,
        policy_factory: Callable[[ExperimentConfig], ReplayablePolicy] | None = None,
    ) -> None:
        """Create a runner wired to `repository` and its collaborators.

        Args:
            repository: The source `ExperienceRepository` every run reads
                from. Never written to by this class — only `.list()` is
                ever called on it.
            reward_calculator: Computes each run's replayed rewards.
                Defaults to a plain `RewardCalculator()`.
            evaluator: Aggregates one run's replay into a `ReplayResult`.
                Defaults to a plain `OfflineEvaluator()`.
            analyzer: Computes cross-run statistics. Defaults to a plain
                `Analyzer()`.
            policy_factory: Builds a fresh policy instance from an
                `ExperimentConfig`. Defaults to `_default_policy_factory`
                (`"HeuristicPolicy"` / `"LinUCBPolicy"`).
        """
        self._repository = repository
        self._reward_calculator = (
            reward_calculator if reward_calculator is not None else RewardCalculator()
        )
        self._evaluator = evaluator if evaluator is not None else OfflineEvaluator()
        self._analyzer = analyzer if analyzer is not None else Analyzer()
        self._policy_factory = (
            policy_factory if policy_factory is not None else _default_policy_factory
        )

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        """Run `config.num_runs` independent runs and aggregate them into an `ExperimentResult`.

        `config.num_runs == 1` replays the source repository's stored
        experiences directly, exactly once. `config.num_runs > 1` draws
        that many independent bootstrap resamples (seeded by
        `config.random_seed`) and replays each one separately — see this
        module's docstring for why. Either way, the source repository and
        every `ExperienceRecord` it holds are read-only throughout.

        Args:
            config: The experiment to run.

        Returns:
            The resulting `ExperimentResult`.
        """
        source_records = self._repository.list()

        if config.num_runs == 1:
            run_results = [self._replay_once(source_records, config)]
        else:
            rng = random.Random(config.random_seed)
            run_results = [
                self._replay_once(self._bootstrap_resample(source_records, rng), config)
                for _ in range(config.num_runs)
            ]

        return self._aggregate(config, run_results)

    def run_sweep(self, configs: list[ExperimentConfig]) -> list[ExperimentResult]:
        """Run `run` for every config in `configs`, e.g. a baseline plus several alpha values.

        Args:
            configs: The experiment configurations to run, in order.

        Returns:
            One `ExperimentResult` per config, in the same order.
        """
        return [self.run(config) for config in configs]

    def _replay_once(
        self, records: list[ExperienceRecord], config: ExperimentConfig
    ) -> ReplayResult:
        """Replay exactly one run's worth of `records` for `config`."""
        repository = BootstrapExperienceRepository(records)
        policy = self._policy_factory(config)
        engine = ReplayEngine(
            repository=repository,
            policy=policy,
            reward_calculator=self._reward_calculator,
            candidate_actions=config.candidate_actions,
        )
        return self._evaluator.evaluate(engine, policy_name=config.policy_name)

    @staticmethod
    def _bootstrap_resample(
        records: list[ExperienceRecord], rng: random.Random
    ) -> list[ExperienceRecord]:
        """Draw `len(records)` samples from `records`, with replacement, using `rng`.

        Args:
            records: The source records to resample from.
            rng: A seeded `random.Random` instance, consumed sequentially
                across every run in one `run()` call so the entire
                sequence of resamples is determined solely by
                `config.random_seed`.

        Returns:
            A new list of the same length as `records` (or `[]` if
            `records` is empty). References the same `ExperienceRecord`
            instances; none are copied or mutated.
        """
        if not records:
            return []
        return [rng.choice(records) for _ in range(len(records))]

    def _aggregate(self, config: ExperimentConfig, runs: list[ReplayResult]) -> ExperimentResult:
        """Fold a list of per-run `ReplayResult`s into one `ExperimentResult`."""
        reward_summary = self._analyzer.summarize([run.average_reward for run in runs])
        quality_values = [run.average_quality for run in runs]
        latency_values = [run.average_latency for run in runs]
        iteration_values = [run.average_iterations for run in runs]
        match_rate_values = [float(run.metadata.get("match_rate", 0.0)) for run in runs]

        critic_names = sorted({name for run in runs for name in run.critic_selection_frequency})
        critic_selection_frequency = {
            name: self._analyzer.mean(
                [run.critic_selection_frequency.get(name, 0.0) for run in runs]
            )
            for name in critic_names
        }

        return ExperimentResult(
            experiment_name=config.experiment_name,
            policy_name=config.policy_name,
            runs=runs,
            average_reward=reward_summary.mean,
            std_reward=reward_summary.std_dev,
            average_quality=self._analyzer.mean(quality_values),
            average_latency=self._analyzer.mean(latency_values),
            average_iterations=self._analyzer.mean(iteration_values),
            match_rate=self._analyzer.mean(match_rate_values),
            critic_selection_frequency=critic_selection_frequency,
            metadata={
                "random_seed": config.random_seed,
                "num_runs": config.num_runs,
                "reward_confidence_interval_95": {
                    "lower": reward_summary.confidence_interval.lower,
                    "upper": reward_summary.confidence_interval.upper,
                },
                "reward_minimum": reward_summary.minimum,
                "reward_maximum": reward_summary.maximum,
            },
        )
