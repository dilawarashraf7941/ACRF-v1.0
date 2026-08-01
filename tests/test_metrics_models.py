"""Unit tests for `ExecutionMetrics` and `ExperimentSummary` (app/metrics/models.py)."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.metrics import ExecutionMetrics, ExperimentSummary


def _make_metrics(**overrides: object) -> ExecutionMetrics:
    defaults: dict[str, object] = {
        "execution_id": "id-1",
        "reward": 0.5,
        "iterations": 0,
        "correction_applied": False,
        "execution_status": "completed",
        "timestamp": datetime.now(timezone.utc),
    }
    defaults.update(overrides)
    return ExecutionMetrics(**defaults)  # type: ignore[arg-type]


# --- ExecutionMetrics ---


def test_execution_metrics_requires_core_fields() -> None:
    with pytest.raises(ValidationError):
        ExecutionMetrics()  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "field",
    ["execution_id", "reward", "iterations", "correction_applied", "execution_status", "timestamp"],
)
def test_execution_metrics_required_fields_are_enforced(field: str) -> None:
    kwargs = {
        "execution_id": "id-1",
        "reward": 0.5,
        "iterations": 0,
        "correction_applied": False,
        "execution_status": "completed",
        "timestamp": datetime.now(timezone.utc),
    }
    del kwargs[field]

    with pytest.raises(ValidationError):
        ExecutionMetrics(**kwargs)  # type: ignore[arg-type]


def test_execution_metrics_applies_defaults() -> None:
    metrics = _make_metrics()

    assert metrics.aggregated_quality_score is None
    assert metrics.latency is None
    assert metrics.estimated_cost is None
    assert metrics.selected_critics == []
    assert metrics.metadata == {}


def test_execution_metrics_accepts_all_fields_explicitly() -> None:
    metrics = _make_metrics(
        aggregated_quality_score=0.7,
        latency=1.5,
        estimated_cost=0.02,
        selected_critics=["LogicCritic", "CodeCritic"],
        correction_applied=True,
        metadata={"policy": "HeuristicPolicy"},
    )

    assert metrics.aggregated_quality_score == 0.7
    assert metrics.latency == 1.5
    assert metrics.estimated_cost == 0.02
    assert metrics.selected_critics == ["LogicCritic", "CodeCritic"]
    assert metrics.correction_applied is True
    assert metrics.metadata == {"policy": "HeuristicPolicy"}


def test_execution_metrics_iterations_must_be_non_negative() -> None:
    with pytest.raises(ValidationError):
        _make_metrics(iterations=-1)


def test_execution_metrics_is_frozen() -> None:
    metrics = _make_metrics()

    with pytest.raises(ValidationError):
        metrics.reward = 99.0  # type: ignore[misc]


def test_execution_metrics_allows_extra_fields() -> None:
    metrics = _make_metrics(custom_field="value")

    assert metrics.custom_field == "value"  # type: ignore[attr-defined]


def test_execution_metrics_round_trips_via_model_dump() -> None:
    metrics = _make_metrics(selected_critics=["LogicCritic"])

    dumped = metrics.model_dump(mode="json")
    reconstructed = ExecutionMetrics(**dumped)

    assert reconstructed == metrics


# --- ExperimentSummary ---


def test_experiment_summary_requires_total_runs() -> None:
    with pytest.raises(ValidationError):
        ExperimentSummary()  # type: ignore[call-arg]


def test_experiment_summary_applies_defaults() -> None:
    summary = ExperimentSummary(total_runs=0)

    assert summary.average_reward is None
    assert summary.average_quality is None
    assert summary.average_iterations is None
    assert summary.average_latency is None
    assert summary.average_cost is None
    assert summary.success_rate is None
    assert summary.correction_rate is None
    assert summary.average_reward_per_policy == {}
    assert summary.critic_selection_frequency == {}
    assert summary.policy_usage == {}
    assert summary.metadata == {}


def test_experiment_summary_total_runs_must_be_non_negative() -> None:
    with pytest.raises(ValidationError):
        ExperimentSummary(total_runs=-1)


@pytest.mark.parametrize("field", ["success_rate", "correction_rate"])
@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_experiment_summary_rates_must_be_within_bounds(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        ExperimentSummary(total_runs=1, **{field: value})


def test_experiment_summary_accepts_all_fields_explicitly() -> None:
    summary = ExperimentSummary(
        total_runs=2,
        average_reward=0.6,
        average_quality=0.7,
        average_iterations=1.0,
        average_latency=0.5,
        average_cost=0.1,
        success_rate=1.0,
        correction_rate=0.5,
        average_reward_per_policy={"HeuristicPolicy": 0.6},
        critic_selection_frequency={"LogicCritic": 2},
        policy_usage={"HeuristicPolicy": 2},
        metadata={"note": "test"},
    )

    assert summary.total_runs == 2
    assert summary.average_reward == 0.6
    assert summary.average_reward_per_policy == {"HeuristicPolicy": 0.6}
    assert summary.critic_selection_frequency == {"LogicCritic": 2}
    assert summary.policy_usage == {"HeuristicPolicy": 2}


def test_experiment_summary_is_frozen() -> None:
    summary = ExperimentSummary(total_runs=1)

    with pytest.raises(ValidationError):
        summary.total_runs = 5  # type: ignore[misc]


def test_experiment_summary_allows_extra_fields() -> None:
    summary = ExperimentSummary(total_runs=0, custom_field="value")

    assert summary.custom_field == "value"  # type: ignore[attr-defined]


def test_experiment_summary_round_trips_via_model_dump() -> None:
    summary = ExperimentSummary(total_runs=1, average_reward=0.5)

    dumped = summary.model_dump(mode="json")
    reconstructed = ExperimentSummary(**dumped)

    assert reconstructed == summary
