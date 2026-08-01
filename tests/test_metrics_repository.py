"""Unit tests for `InMemoryMetricsRepository` (app/metrics/repository.py)."""

from datetime import datetime, timezone

import pytest

from app.metrics import (
    ExecutionMetrics,
    ExperimentSummary,
    InMemoryMetricsRepository,
    MetricsRepository,
)


def _make_metrics(execution_id: str = "id-1", **overrides: object) -> ExecutionMetrics:
    defaults: dict[str, object] = {
        "execution_id": execution_id,
        "reward": 0.5,
        "iterations": 0,
        "correction_applied": False,
        "execution_status": "completed",
        "timestamp": datetime.now(timezone.utc),
    }
    defaults.update(overrides)
    return ExecutionMetrics(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def repository() -> InMemoryMetricsRepository:
    return InMemoryMetricsRepository()


def test_new_repository_is_empty(repository: InMemoryMetricsRepository) -> None:
    assert repository.count() == 0
    assert repository.list() == []


def test_add_then_count(repository: InMemoryMetricsRepository) -> None:
    repository.add(_make_metrics())

    assert repository.count() == 1


def test_add_multiple_records(repository: InMemoryMetricsRepository) -> None:
    repository.add(_make_metrics("id-1"))
    repository.add(_make_metrics("id-2"))
    repository.add(_make_metrics("id-3"))

    assert repository.count() == 3


def test_add_allows_repeated_execution_ids() -> None:
    """Unlike ExperienceRepository, MetricsRepository has no `get()` and
    does not need unique keys — repeated ids are simply appended.
    """
    repository = InMemoryMetricsRepository()
    repository.add(_make_metrics("same-id"))
    repository.add(_make_metrics("same-id"))

    assert repository.count() == 2


def test_list_returns_all_records(repository: InMemoryMetricsRepository) -> None:
    metrics_a = _make_metrics("id-1")
    metrics_b = _make_metrics("id-2")
    repository.add(metrics_a)
    repository.add(metrics_b)

    listed = repository.list()

    assert len(listed) == 2
    assert metrics_a in listed
    assert metrics_b in listed


def test_list_preserves_insertion_order(repository: InMemoryMetricsRepository) -> None:
    repository.add(_make_metrics("id-3"))
    repository.add(_make_metrics("id-1"))
    repository.add(_make_metrics("id-2"))

    assert [m.execution_id for m in repository.list()] == ["id-3", "id-1", "id-2"]


def test_list_returns_a_copy_not_internal_state(repository: InMemoryMetricsRepository) -> None:
    repository.add(_make_metrics("id-1"))

    listed = repository.list()
    listed.append(_make_metrics("id-2"))

    assert repository.count() == 1


def test_clear_empties_the_repository(repository: InMemoryMetricsRepository) -> None:
    repository.add(_make_metrics("id-1"))
    repository.add(_make_metrics("id-2"))

    repository.clear()

    assert repository.count() == 0
    assert repository.list() == []


def test_clear_on_empty_repository_is_a_no_op(repository: InMemoryMetricsRepository) -> None:
    repository.clear()

    assert repository.count() == 0


def test_add_after_clear_works_again(repository: InMemoryMetricsRepository) -> None:
    repository.add(_make_metrics("id-1"))
    repository.clear()
    repository.add(_make_metrics("id-1"))

    assert repository.count() == 1


def test_count_reflects_current_size(repository: InMemoryMetricsRepository) -> None:
    assert repository.count() == 0
    repository.add(_make_metrics("id-1"))
    assert repository.count() == 1
    repository.add(_make_metrics("id-2"))
    assert repository.count() == 2


# --- summary() ---


def test_summary_on_empty_repository_returns_zero_runs(
    repository: InMemoryMetricsRepository,
) -> None:
    summary = repository.summary()

    assert isinstance(summary, ExperimentSummary)
    assert summary.total_runs == 0
    assert summary.average_reward is None


def test_summary_reflects_stored_records(repository: InMemoryMetricsRepository) -> None:
    repository.add(_make_metrics("id-1", reward=0.2))
    repository.add(_make_metrics("id-2", reward=0.8))

    summary = repository.summary()

    assert summary.total_runs == 2
    assert summary.average_reward == pytest.approx(0.5)


def test_summary_reflects_clear(repository: InMemoryMetricsRepository) -> None:
    repository.add(_make_metrics("id-1"))
    repository.clear()

    summary = repository.summary()

    assert summary.total_runs == 0


def test_summary_uses_injected_aggregator() -> None:
    class StubAggregator:
        def aggregate(self, metrics: list[ExecutionMetrics]) -> ExperimentSummary:
            return ExperimentSummary(total_runs=999)

    repository = InMemoryMetricsRepository(aggregator=StubAggregator())  # type: ignore[arg-type]
    repository.add(_make_metrics())

    assert repository.summary().total_runs == 999


# --- Interface conformance ---


def test_is_a_metrics_repository() -> None:
    repository = InMemoryMetricsRepository()

    assert isinstance(repository, MetricsRepository)


def test_metrics_repository_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        MetricsRepository()  # type: ignore[abstract]
