"""Unit tests for `MetricsAggregator` (app/metrics/aggregator.py)."""

from datetime import datetime, timezone

import pytest

from app.metrics import ExecutionMetrics, ExperimentSummary, MetricsAggregator


def _make_metrics(**overrides: object) -> ExecutionMetrics:
    defaults: dict[str, object] = {
        "execution_id": "id-1",
        "reward": 0.5,
        "iterations": 0,
        "correction_applied": False,
        "execution_status": "completed",
        "timestamp": datetime.now(timezone.utc),
        "metadata": {"policy": "HeuristicPolicy"},
    }
    defaults.update(overrides)
    return ExecutionMetrics(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def aggregator() -> MetricsAggregator:
    return MetricsAggregator()


# --- Empty repository ---


def test_aggregate_empty_list_returns_zero_runs(aggregator: MetricsAggregator) -> None:
    summary = aggregator.aggregate([])

    assert isinstance(summary, ExperimentSummary)
    assert summary.total_runs == 0


def test_aggregate_empty_list_returns_none_averages(aggregator: MetricsAggregator) -> None:
    summary = aggregator.aggregate([])

    assert summary.average_reward is None
    assert summary.average_quality is None
    assert summary.average_iterations is None
    assert summary.average_latency is None
    assert summary.average_cost is None
    assert summary.success_rate is None
    assert summary.correction_rate is None


def test_aggregate_empty_list_returns_empty_dicts(aggregator: MetricsAggregator) -> None:
    summary = aggregator.aggregate([])

    assert summary.average_reward_per_policy == {}
    assert summary.critic_selection_frequency == {}
    assert summary.policy_usage == {}


def test_aggregate_empty_list_does_not_raise(aggregator: MetricsAggregator) -> None:
    aggregator.aggregate([])  # must not raise


# --- total_runs / averages ---


def test_total_runs_matches_input_length(aggregator: MetricsAggregator) -> None:
    metrics = [_make_metrics(execution_id=str(i)) for i in range(4)]

    summary = aggregator.aggregate(metrics)

    assert summary.total_runs == 4


def test_average_reward_is_computed_correctly(aggregator: MetricsAggregator) -> None:
    metrics = [_make_metrics(reward=0.2), _make_metrics(reward=0.8)]

    summary = aggregator.aggregate(metrics)

    assert summary.average_reward == pytest.approx(0.5)


def test_average_iterations_is_computed_correctly(aggregator: MetricsAggregator) -> None:
    metrics = [_make_metrics(iterations=0), _make_metrics(iterations=4)]

    summary = aggregator.aggregate(metrics)

    assert summary.average_iterations == pytest.approx(2.0)


def test_average_quality_excludes_none_values(aggregator: MetricsAggregator) -> None:
    metrics = [
        _make_metrics(aggregated_quality_score=0.8),
        _make_metrics(aggregated_quality_score=None),
        _make_metrics(aggregated_quality_score=0.4),
    ]

    summary = aggregator.aggregate(metrics)

    assert summary.average_quality == pytest.approx(0.6)


def test_average_quality_is_none_when_all_values_missing(aggregator: MetricsAggregator) -> None:
    metrics = [_make_metrics(aggregated_quality_score=None) for _ in range(3)]

    summary = aggregator.aggregate(metrics)

    assert summary.average_quality is None


def test_average_latency_excludes_none_values(aggregator: MetricsAggregator) -> None:
    metrics = [
        _make_metrics(latency=1.0),
        _make_metrics(latency=None),
        _make_metrics(latency=3.0),
    ]

    summary = aggregator.aggregate(metrics)

    assert summary.average_latency == pytest.approx(2.0)


def test_average_cost_excludes_none_values(aggregator: MetricsAggregator) -> None:
    metrics = [
        _make_metrics(estimated_cost=0.1),
        _make_metrics(estimated_cost=None),
        _make_metrics(estimated_cost=0.3),
    ]

    summary = aggregator.aggregate(metrics)

    assert summary.average_cost == pytest.approx(0.2)


# --- success_rate / correction_rate ---


def test_success_rate_is_fraction_of_completed_runs(aggregator: MetricsAggregator) -> None:
    metrics = [
        _make_metrics(execution_status="completed"),
        _make_metrics(execution_status="completed"),
        _make_metrics(execution_status="failed"),
        _make_metrics(execution_status="failed"),
    ]

    summary = aggregator.aggregate(metrics)

    assert summary.success_rate == pytest.approx(0.5)


def test_success_rate_is_one_when_all_completed(aggregator: MetricsAggregator) -> None:
    metrics = [_make_metrics(execution_status="completed") for _ in range(3)]

    summary = aggregator.aggregate(metrics)

    assert summary.success_rate == 1.0


def test_success_rate_is_zero_when_none_completed(aggregator: MetricsAggregator) -> None:
    metrics = [_make_metrics(execution_status="failed") for _ in range(3)]

    summary = aggregator.aggregate(metrics)

    assert summary.success_rate == 0.0


def test_correction_rate_is_fraction_of_runs_with_correction_applied(
    aggregator: MetricsAggregator,
) -> None:
    metrics = [
        _make_metrics(correction_applied=True),
        _make_metrics(correction_applied=True),
        _make_metrics(correction_applied=False),
        _make_metrics(correction_applied=False),
    ]

    summary = aggregator.aggregate(metrics)

    assert summary.correction_rate == pytest.approx(0.5)


# --- critic_selection_frequency ---


def test_critic_selection_frequency_counts_across_all_runs(aggregator: MetricsAggregator) -> None:
    metrics = [
        _make_metrics(selected_critics=["LogicCritic", "CodeCritic"]),
        _make_metrics(selected_critics=["LogicCritic"]),
        _make_metrics(selected_critics=["FactCritic"]),
    ]

    summary = aggregator.aggregate(metrics)

    assert summary.critic_selection_frequency == {
        "LogicCritic": 2,
        "CodeCritic": 1,
        "FactCritic": 1,
    }


def test_critic_selection_frequency_empty_when_no_critics_selected(
    aggregator: MetricsAggregator,
) -> None:
    metrics = [_make_metrics(selected_critics=[])]

    summary = aggregator.aggregate(metrics)

    assert summary.critic_selection_frequency == {}


# --- policy_usage / average_reward_per_policy ---


def test_policy_usage_counts_runs_per_policy(aggregator: MetricsAggregator) -> None:
    metrics = [
        _make_metrics(metadata={"policy": "HeuristicPolicy"}),
        _make_metrics(metadata={"policy": "HeuristicPolicy"}),
        _make_metrics(metadata={"policy": "ContextualBandit"}),
    ]

    summary = aggregator.aggregate(metrics)

    assert summary.policy_usage == {"HeuristicPolicy": 2, "ContextualBandit": 1}


def test_policy_usage_uses_unknown_label_when_policy_missing(aggregator: MetricsAggregator) -> None:
    metrics = [_make_metrics(metadata={})]

    summary = aggregator.aggregate(metrics)

    assert summary.policy_usage == {"unknown": 1}


def test_average_reward_per_policy_groups_correctly(aggregator: MetricsAggregator) -> None:
    metrics = [
        _make_metrics(reward=0.2, metadata={"policy": "HeuristicPolicy"}),
        _make_metrics(reward=0.8, metadata={"policy": "HeuristicPolicy"}),
        _make_metrics(reward=1.0, metadata={"policy": "ContextualBandit"}),
    ]

    summary = aggregator.aggregate(metrics)

    assert summary.average_reward_per_policy["HeuristicPolicy"] == pytest.approx(0.5)
    assert summary.average_reward_per_policy["ContextualBandit"] == pytest.approx(1.0)


def test_average_reward_per_policy_is_independent_of_overall_average(
    aggregator: MetricsAggregator,
) -> None:
    metrics = [
        _make_metrics(reward=0.0, metadata={"policy": "PolicyA"}),
        _make_metrics(reward=1.0, metadata={"policy": "PolicyB"}),
    ]

    summary = aggregator.aggregate(metrics)

    assert summary.average_reward == pytest.approx(0.5)
    assert summary.average_reward_per_policy == {"PolicyA": 0.0, "PolicyB": 1.0}


# --- Metadata diagnostics ---


def test_metadata_records_counts_of_available_optional_fields(
    aggregator: MetricsAggregator,
) -> None:
    metrics = [
        _make_metrics(aggregated_quality_score=0.5, latency=1.0, estimated_cost=None),
        _make_metrics(aggregated_quality_score=None, latency=None, estimated_cost=0.1),
    ]

    summary = aggregator.aggregate(metrics)

    assert summary.metadata["runs_with_quality_score"] == 1
    assert summary.metadata["runs_with_latency"] == 1
    assert summary.metadata["runs_with_cost"] == 1


# --- Deterministic calculations ---


def test_aggregate_is_deterministic(aggregator: MetricsAggregator) -> None:
    metrics = [
        _make_metrics(reward=0.3, execution_id="a"),
        _make_metrics(reward=0.7, execution_id="b"),
    ]

    summary_a = aggregator.aggregate(list(metrics))
    summary_b = aggregator.aggregate(list(metrics))

    assert summary_a == summary_b


def test_aggregate_does_not_mutate_input_list(aggregator: MetricsAggregator) -> None:
    metrics = [_make_metrics(execution_id="a"), _make_metrics(execution_id="b")]
    original = list(metrics)

    aggregator.aggregate(metrics)

    assert metrics == original
