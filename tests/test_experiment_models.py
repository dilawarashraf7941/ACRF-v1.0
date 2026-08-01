"""Unit tests for the Experiment Framework's data models.

See `app/evaluation/experiments/models.py`.
"""

import pytest
from pydantic import ValidationError

from app.evaluation.experiments.models import (
    ConfidenceInterval,
    ExperimentConfig,
    ExperimentResult,
    StatisticalSummary,
)
from app.evaluation.offline.models import ReplayResult


def _make_replay_result(**overrides: object) -> ReplayResult:
    defaults: dict[str, object] = {
        "policy_name": "HeuristicPolicy",
        "total_experiences": 2,
        "total_reward": 1.0,
        "average_reward": 0.5,
        "average_quality": 0.6,
        "average_iterations": 1.0,
        "average_latency": 1.5,
    }
    defaults.update(overrides)
    return ReplayResult(**defaults)


# --- ExperimentConfig ---


def test_experiment_config_constructs_with_required_fields() -> None:
    config = ExperimentConfig(
        experiment_name="baseline", policy_name="HeuristicPolicy", random_seed=1, num_runs=1
    )
    assert config.alpha is None
    assert config.candidate_actions is None
    assert config.metadata == {}


def test_experiment_config_missing_required_field_raises() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig(policy_name="HeuristicPolicy", random_seed=1, num_runs=1)


def test_experiment_config_num_runs_must_be_at_least_one() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig(
            experiment_name="baseline", policy_name="HeuristicPolicy", random_seed=1, num_runs=0
        )


def test_experiment_config_is_frozen() -> None:
    config = ExperimentConfig(
        experiment_name="baseline", policy_name="HeuristicPolicy", random_seed=1, num_runs=1
    )
    with pytest.raises(ValidationError):
        config.num_runs = 5  # type: ignore[misc]


def test_experiment_config_holds_alpha_and_candidate_actions() -> None:
    config = ExperimentConfig(
        experiment_name="candidate",
        policy_name="LinUCBPolicy",
        alpha=0.5,
        random_seed=1,
        num_runs=1,
        candidate_actions=["CodeCritic", "LogicCritic"],
    )
    assert config.alpha == 0.5
    assert config.candidate_actions == ["CodeCritic", "LogicCritic"]


# --- ConfidenceInterval ---


def test_confidence_interval_defaults_to_95_percent() -> None:
    interval = ConfidenceInterval(lower=0.1, upper=0.9)
    assert interval.confidence_level == 0.95


def test_confidence_interval_level_must_be_within_bounds() -> None:
    with pytest.raises(ValidationError):
        ConfidenceInterval(lower=0.1, upper=0.9, confidence_level=1.5)


def test_confidence_interval_is_frozen() -> None:
    interval = ConfidenceInterval(lower=0.1, upper=0.9)
    with pytest.raises(ValidationError):
        interval.lower = 0.2  # type: ignore[misc]


# --- StatisticalSummary ---


def test_statistical_summary_constructs_with_confidence_interval() -> None:
    summary = StatisticalSummary(
        mean=0.5,
        std_dev=0.1,
        minimum=0.3,
        maximum=0.7,
        confidence_interval=ConfidenceInterval(lower=0.4, upper=0.6),
        sample_size=10,
    )
    assert summary.sample_size == 10
    assert summary.confidence_interval.lower == 0.4


def test_statistical_summary_negative_sample_size_raises() -> None:
    with pytest.raises(ValidationError):
        StatisticalSummary(
            mean=0.5,
            std_dev=0.1,
            minimum=0.3,
            maximum=0.7,
            confidence_interval=ConfidenceInterval(lower=0.4, upper=0.6),
            sample_size=-1,
        )


# --- ExperimentResult ---


def test_experiment_result_constructs_with_required_fields() -> None:
    result = ExperimentResult(
        experiment_name="baseline",
        policy_name="HeuristicPolicy",
        average_reward=0.5,
        std_reward=0.1,
        average_quality=0.6,
        average_latency=1.0,
        average_iterations=1.0,
        match_rate=0.5,
    )
    assert result.runs == []
    assert result.critic_selection_frequency == {}
    assert result.metadata == {}


def test_experiment_result_holds_runs() -> None:
    replay_result = _make_replay_result()
    result = ExperimentResult(
        experiment_name="baseline",
        policy_name="HeuristicPolicy",
        runs=[replay_result],
        average_reward=0.5,
        std_reward=0.1,
        average_quality=0.6,
        average_latency=1.0,
        average_iterations=1.0,
        match_rate=0.5,
    )
    assert result.runs == [replay_result]


def test_experiment_result_missing_required_field_raises() -> None:
    with pytest.raises(ValidationError):
        ExperimentResult(experiment_name="baseline", policy_name="HeuristicPolicy")


def test_experiment_result_is_frozen() -> None:
    result = ExperimentResult(
        experiment_name="baseline",
        policy_name="HeuristicPolicy",
        average_reward=0.5,
        std_reward=0.1,
        average_quality=0.6,
        average_latency=1.0,
        average_iterations=1.0,
        match_rate=0.5,
    )
    with pytest.raises(ValidationError):
        result.average_reward = 1.0  # type: ignore[misc]
