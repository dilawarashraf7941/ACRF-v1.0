"""Unit tests for `ExperimentRunner` (`app/evaluation/experiments/runner.py`).

Covers: single run, multiple runs, determinism, aggregation,
`HeuristicPolicy`/`LinUCBPolicy` support, `run_sweep`, dependency
injection (custom policy_factory/evaluator/analyzer), and the Research
Constraints (no mutation of source data, no policy updates during a run).
"""

from datetime import datetime, timezone

import pytest

from app.evaluation.experiments import ExperimentConfig, ExperimentRunner
from app.evaluation.experiments.runner import BootstrapExperienceRepository, _default_policy_factory
from app.experience import ExperienceRecord, InMemoryExperienceRepository
from app.policy import HeuristicPolicy
from app.policy.linucb import LinUCBPolicy

_TIMESTAMP = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _make_experience(
    experience_id: str,
    selected_critics: list[str],
    quality: float = 0.7,
    iterations: int = 1,
    latency: float = 1.0,
) -> ExperienceRecord:
    return ExperienceRecord(
        experience_id=experience_id,
        session_id="session-1",
        task_id="task-1",
        timestamp=_TIMESTAMP,
        state_features={"error_feature_count": 0, "worker_output_count": 1},
        selected_critics=selected_critics,
        critic_scores={critic: 0.7 for critic in selected_critics},
        aggregated_quality_score=quality,
        iterations=iterations,
        execution_status="completed",
        latency=latency,
    )


def _make_repository(size: int = 8) -> InMemoryExperienceRepository:
    repository = InMemoryExperienceRepository()
    pattern = [["CodeCritic"], ["LogicCritic"], ["CodeCritic"], ["FactCritic", "MetaCritic"]]
    for i in range(size):
        critics = pattern[i % len(pattern)]
        repository.add(
            _make_experience(
                f"exp-{i}", critics, quality=0.5 + 0.05 * i, iterations=i % 4, latency=1.0 + 0.1 * i
            )
        )
    return repository


# --- BootstrapExperienceRepository ---


def test_bootstrap_repository_allows_duplicate_ids() -> None:
    record = _make_experience("dup", ["CodeCritic"])
    repository = BootstrapExperienceRepository([record, record, record])

    assert repository.count() == 3
    assert repository.list() == [record, record, record]


def test_bootstrap_repository_get_returns_first_match() -> None:
    record = _make_experience("dup", ["CodeCritic"])
    repository = BootstrapExperienceRepository([record, record])

    assert repository.get("dup") is record
    assert repository.get("missing") is None


def test_bootstrap_repository_add_appends_without_uniqueness_check() -> None:
    repository = BootstrapExperienceRepository()
    record = _make_experience("exp-1", ["CodeCritic"])

    repository.add(record)
    repository.add(record)

    assert repository.count() == 2


def test_bootstrap_repository_clear() -> None:
    repository = BootstrapExperienceRepository([_make_experience("exp-1", ["CodeCritic"])])
    repository.clear()
    assert repository.count() == 0


# --- default policy factory ---


def test_default_policy_factory_builds_heuristic_policy() -> None:
    config = ExperimentConfig(
        experiment_name="e", policy_name="HeuristicPolicy", random_seed=1, num_runs=1
    )
    assert isinstance(_default_policy_factory(config), HeuristicPolicy)


def test_default_policy_factory_builds_linucb_policy_with_alpha() -> None:
    config = ExperimentConfig(
        experiment_name="e", policy_name="LinUCBPolicy", alpha=0.5, random_seed=1, num_runs=1
    )
    policy = _default_policy_factory(config)
    assert isinstance(policy, LinUCBPolicy)
    assert policy.alpha == 0.5


def test_default_policy_factory_linucb_defaults_alpha_to_one() -> None:
    config = ExperimentConfig(
        experiment_name="e", policy_name="LinUCBPolicy", random_seed=1, num_runs=1
    )
    policy = _default_policy_factory(config)
    assert isinstance(policy, LinUCBPolicy)
    assert policy.alpha == 1.0


def test_default_policy_factory_rejects_unrecognized_policy_name() -> None:
    config = ExperimentConfig(
        experiment_name="e", policy_name="NoSuchPolicy", random_seed=1, num_runs=1
    )
    with pytest.raises(ValueError, match="Unrecognized policy_name"):
        _default_policy_factory(config)


# --- single run ---


def test_single_run_produces_exactly_one_run_result() -> None:
    repository = _make_repository()
    runner = ExperimentRunner(repository=repository)
    config = ExperimentConfig(
        experiment_name="baseline", policy_name="HeuristicPolicy", random_seed=1, num_runs=1
    )

    result = runner.run(config)

    assert len(result.runs) == 1
    assert result.experiment_name == "baseline"
    assert result.policy_name == "HeuristicPolicy"


def test_single_run_std_reward_is_zero() -> None:
    repository = _make_repository()
    runner = ExperimentRunner(repository=repository)
    config = ExperimentConfig(
        experiment_name="baseline", policy_name="HeuristicPolicy", random_seed=1, num_runs=1
    )

    result = runner.run(config)

    assert result.std_reward == 0.0
    assert result.average_reward == result.runs[0].average_reward


def test_single_run_replays_source_data_directly_without_resampling() -> None:
    repository = _make_repository(size=6)
    runner = ExperimentRunner(repository=repository)
    config = ExperimentConfig(
        experiment_name="baseline", policy_name="HeuristicPolicy", random_seed=1, num_runs=1
    )

    result = runner.run(config)

    assert result.runs[0].metadata["total_stored_experiences"] == 6


# --- multiple runs ---


def test_multiple_runs_produce_num_runs_results() -> None:
    repository = _make_repository()
    runner = ExperimentRunner(repository=repository)
    config = ExperimentConfig(
        experiment_name="baseline", policy_name="HeuristicPolicy", random_seed=1, num_runs=15
    )

    result = runner.run(config)

    assert len(result.runs) == 15


def test_multiple_runs_bootstrap_resample_is_same_size_as_source() -> None:
    repository = _make_repository(size=8)
    runner = ExperimentRunner(repository=repository)
    config = ExperimentConfig(
        experiment_name="baseline", policy_name="HeuristicPolicy", random_seed=1, num_runs=5
    )

    result = runner.run(config)

    for run in result.runs:
        assert run.metadata["total_stored_experiences"] == 8


def test_multiple_runs_produce_genuine_variation_via_resampling() -> None:
    repository = _make_repository(size=8)
    runner = ExperimentRunner(repository=repository)
    config = ExperimentConfig(
        experiment_name="baseline", policy_name="HeuristicPolicy", random_seed=1, num_runs=25
    )

    result = runner.run(config)

    reward_values = {run.average_reward for run in result.runs}
    assert len(reward_values) > 1, (
        "bootstrap resampling should produce more than one distinct value"
    )


def test_multiple_runs_does_not_error_on_empty_repository() -> None:
    repository = InMemoryExperienceRepository()
    runner = ExperimentRunner(repository=repository)
    config = ExperimentConfig(
        experiment_name="baseline", policy_name="HeuristicPolicy", random_seed=1, num_runs=5
    )

    result = runner.run(config)

    assert len(result.runs) == 5
    assert result.average_reward == 0.0


# --- determinism ---


def test_run_is_deterministic_given_the_same_config_and_seed() -> None:
    repository = _make_repository(size=10)
    runner = ExperimentRunner(repository=repository)
    config = ExperimentConfig(
        experiment_name="baseline", policy_name="HeuristicPolicy", random_seed=99, num_runs=12
    )

    result_a = runner.run(config)
    result_b = runner.run(config)

    assert result_a == result_b


def test_run_with_different_seed_produces_different_resamples() -> None:
    repository = _make_repository(size=10)
    runner = ExperimentRunner(repository=repository)
    config_a = ExperimentConfig(
        experiment_name="baseline", policy_name="HeuristicPolicy", random_seed=1, num_runs=8
    )
    config_b = ExperimentConfig(
        experiment_name="baseline", policy_name="HeuristicPolicy", random_seed=2, num_runs=8
    )

    result_a = runner.run(config_a)
    result_b = runner.run(config_b)

    assert [run.average_reward for run in result_a.runs] != [
        run.average_reward for run in result_b.runs
    ]


def test_two_independent_runners_with_same_config_agree() -> None:
    config = ExperimentConfig(
        experiment_name="baseline", policy_name="HeuristicPolicy", random_seed=5, num_runs=10
    )

    runner_a = ExperimentRunner(repository=_make_repository(size=8))
    runner_b = ExperimentRunner(repository=_make_repository(size=8))

    assert runner_a.run(config) == runner_b.run(config)


# --- aggregation ---


def test_aggregation_average_reward_is_mean_of_run_rewards() -> None:
    repository = _make_repository(size=8)
    runner = ExperimentRunner(repository=repository)
    config = ExperimentConfig(
        experiment_name="baseline", policy_name="HeuristicPolicy", random_seed=3, num_runs=10
    )

    result = runner.run(config)

    expected_mean = sum(run.average_reward for run in result.runs) / len(result.runs)
    assert result.average_reward == pytest.approx(expected_mean)


def test_aggregation_records_confidence_interval_in_metadata() -> None:
    repository = _make_repository(size=8)
    runner = ExperimentRunner(repository=repository)
    config = ExperimentConfig(
        experiment_name="baseline", policy_name="HeuristicPolicy", random_seed=3, num_runs=10
    )

    result = runner.run(config)

    interval = result.metadata["reward_confidence_interval_95"]
    assert interval["lower"] <= result.average_reward <= interval["upper"]
    assert result.metadata["random_seed"] == 3
    assert result.metadata["num_runs"] == 10


def test_aggregation_critic_selection_frequency_averages_across_runs() -> None:
    repository = _make_repository(size=8)
    runner = ExperimentRunner(repository=repository)
    config = ExperimentConfig(
        experiment_name="baseline", policy_name="HeuristicPolicy", random_seed=3, num_runs=10
    )

    result = runner.run(config)

    for frequency in result.critic_selection_frequency.values():
        assert 0.0 <= frequency <= 1.0


def test_aggregation_match_rate_is_mean_of_run_match_rates() -> None:
    repository = _make_repository(size=8)
    runner = ExperimentRunner(repository=repository)
    config = ExperimentConfig(
        experiment_name="baseline", policy_name="HeuristicPolicy", random_seed=3, num_runs=6
    )

    result = runner.run(config)

    expected = sum(float(run.metadata.get("match_rate", 0.0)) for run in result.runs) / 6
    assert result.match_rate == pytest.approx(expected)


# --- HeuristicPolicy / LinUCBPolicy support ---


def test_supports_heuristic_policy() -> None:
    repository = _make_repository()
    runner = ExperimentRunner(repository=repository)
    config = ExperimentConfig(
        experiment_name="baseline", policy_name="HeuristicPolicy", random_seed=1, num_runs=1
    )
    result = runner.run(config)
    assert result.policy_name == "HeuristicPolicy"
    assert all(run.policy_name == "HeuristicPolicy" for run in result.runs)


def test_supports_linucb_policy() -> None:
    repository = _make_repository()
    runner = ExperimentRunner(repository=repository)
    config = ExperimentConfig(
        experiment_name="candidate",
        policy_name="LinUCBPolicy",
        alpha=0.5,
        random_seed=1,
        num_runs=1,
    )
    result = runner.run(config)
    assert result.policy_name == "LinUCBPolicy"
    assert all(run.policy_name == "LinUCBPolicy" for run in result.runs)


# --- run_sweep ---


def test_run_sweep_runs_every_config_in_order() -> None:
    repository = _make_repository()
    runner = ExperimentRunner(repository=repository)
    configs = [
        ExperimentConfig(
            experiment_name=f"linucb-alpha-{alpha}",
            policy_name="LinUCBPolicy",
            alpha=alpha,
            random_seed=1,
            num_runs=1,
        )
        for alpha in (0.25, 0.5, 1.0, 2.0)
    ]

    results = runner.run_sweep(configs)

    assert [r.experiment_name for r in results] == [
        "linucb-alpha-0.25",
        "linucb-alpha-0.5",
        "linucb-alpha-1.0",
        "linucb-alpha-2.0",
    ]


# --- dependency injection ---


def test_custom_policy_factory_is_used() -> None:
    repository = _make_repository()
    trained_policy = LinUCBPolicy(alpha=0.3)
    warmup_context_experience = repository.list()[0]

    calls: list[str] = []

    def factory(config: ExperimentConfig) -> LinUCBPolicy:
        calls.append(config.policy_name)
        return trained_policy

    runner = ExperimentRunner(repository=repository, policy_factory=factory)
    config = ExperimentConfig(
        experiment_name="custom", policy_name="AnythingGoes", random_seed=1, num_runs=3
    )

    result = runner.run(config)

    assert calls == ["AnythingGoes", "AnythingGoes", "AnythingGoes"]
    assert len(result.runs) == 3
    del warmup_context_experience  # unused; kept only to document intent


def test_custom_analyzer_is_used() -> None:
    from app.evaluation.experiments.analyzer import Analyzer

    class _StubAnalyzer(Analyzer):
        def summarize(self, values, confidence_level=0.95):  # type: ignore[override]
            summary = super().summarize(values, confidence_level)
            return summary.model_copy(update={"mean": 999.0})

    repository = _make_repository()
    runner = ExperimentRunner(repository=repository, analyzer=_StubAnalyzer())
    config = ExperimentConfig(
        experiment_name="baseline", policy_name="HeuristicPolicy", random_seed=1, num_runs=2
    )

    result = runner.run(config)

    assert result.average_reward == 999.0


# --- Research Constraints ---


def test_run_does_not_mutate_source_repository() -> None:
    repository = _make_repository(size=6)
    dumps_before = {record.experience_id: record.model_dump() for record in repository.list()}
    runner = ExperimentRunner(repository=repository)
    config = ExperimentConfig(
        experiment_name="baseline", policy_name="HeuristicPolicy", random_seed=1, num_runs=10
    )

    runner.run(config)

    assert repository.count() == 6
    for record in repository.list():
        assert record.model_dump() == dumps_before[record.experience_id]


def test_run_never_calls_update_on_linucb_policy() -> None:
    repository = _make_repository(size=8)
    built_policies: list[LinUCBPolicy] = []

    def tracking_factory(config: ExperimentConfig) -> LinUCBPolicy:
        policy = LinUCBPolicy(alpha=config.alpha or 1.0)
        built_policies.append(policy)
        return policy

    runner = ExperimentRunner(repository=repository, policy_factory=tracking_factory)
    config = ExperimentConfig(
        experiment_name="candidate",
        policy_name="LinUCBPolicy",
        alpha=1.0,
        random_seed=1,
        num_runs=5,
    )

    runner.run(config)

    assert len(built_policies) == 5
    for policy in built_policies:
        for arm in policy.arms.values():
            assert (arm.b == 0.0).all(), "no arm should have been updated during a replay-only run"
