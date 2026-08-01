"""Storage layer for `ExecutionMetrics`.

Defines an abstract `MetricsRepository` interface and one concrete,
in-memory-only implementation. No persistence, no database — this module
only stores records for the lifetime of the Python process. The abstract
interface exists specifically so a future SQLite/ChromaDB/PostgreSQL-backed
implementation can be substituted without requiring any change to
`MetricsCollector`, `MetricsAggregator`, or any other code that depends on
`MetricsRepository`.
"""

from abc import ABC, abstractmethod

from app.metrics.aggregator import MetricsAggregator
from app.metrics.models import ExecutionMetrics, ExperimentSummary


class MetricsRepository(ABC):
    """Abstract interface for storing, retrieving, and summarizing `ExecutionMetrics`.

    Any concrete implementation (in-memory, SQLite, ChromaDB, PostgreSQL,
    ...) must provide these five operations. Code that depends on this
    interface — including `MetricsCollector` — never needs to know which
    concrete implementation it is talking to.
    """

    @abstractmethod
    def add(self, metrics: ExecutionMetrics) -> None:
        """Store `metrics`.

        Args:
            metrics: The execution metrics record to store.
        """
        raise NotImplementedError

    @abstractmethod
    def list(self) -> list[ExecutionMetrics]:
        """Return every stored record.

        Returns:
            All stored `ExecutionMetrics`, in a stable, documented order
            (this implementation returns insertion order).
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Remove every stored record."""
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored records."""
        raise NotImplementedError

    @abstractmethod
    def summary(self) -> ExperimentSummary:
        """Return an `ExperimentSummary` aggregated over every stored record."""
        raise NotImplementedError


class InMemoryMetricsRepository(MetricsRepository):
    """An in-memory-only `MetricsRepository`.

    Records live only in a plain Python list for the lifetime of this
    instance; nothing is written to disk or to a database. `summary()`
    delegates to an injected `MetricsAggregator` (dependency injection),
    defaulting to a plain `MetricsAggregator()` when none is supplied.
    """

    def __init__(self, aggregator: MetricsAggregator | None = None) -> None:
        """Create an empty repository, optionally wired to a specific aggregator.

        Args:
            aggregator: The `MetricsAggregator` `summary()` delegates to.
                Defaults to a plain `MetricsAggregator()` when not provided.
        """
        self._metrics: list[ExecutionMetrics] = []
        self._aggregator = aggregator if aggregator is not None else MetricsAggregator()

    def add(self, metrics: ExecutionMetrics) -> None:
        """Append `metrics` to this repository."""
        self._metrics.append(metrics)

    def list(self) -> list[ExecutionMetrics]:
        """Return every stored record, in insertion order."""
        return list(self._metrics)

    def clear(self) -> None:
        """Remove every stored record."""
        self._metrics.clear()

    def count(self) -> int:
        """Return the number of stored records."""
        return len(self._metrics)

    def summary(self) -> ExperimentSummary:
        """Return an `ExperimentSummary` aggregated over every currently stored record."""
        return self._aggregator.aggregate(self.list())


DEFAULT_METRICS_REPOSITORY: MetricsRepository = InMemoryMetricsRepository()
"""A process-wide, shared `InMemoryMetricsRepository` instance.

`evaluation_node` (see `app/graph/nodes.py`) is a plain function invoked
by LangGraph with only `(state)` — there is no constructor or call site
available to inject a repository into it from outside. This module-level
singleton is the practical mechanism by which `evaluation_node` obtains a
repository that persists across calls within a process, mirroring
`DEFAULT_EXPERIENCE_REPOSITORY` (see `app/experience/repository.py`).
Tests should construct and inject their own `InMemoryMetricsRepository`
instead of relying on this shared singleton, to stay isolated from other
tests.
"""
