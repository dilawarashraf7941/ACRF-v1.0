"""Critic aggregation infrastructure for ACRF: the abstract aggregation
strategy interface and four placeholder strategy implementations.

This module contains no real aggregation algorithms — no vote counting,
no weighted averaging, no max-score selection, and no policy-weighted
combination. Every concrete strategy's `aggregate` method returns a
fixed, valid, neutral `AggregatedCriticResult`; it echoes back which
critics contributed (structural bookkeeping only, exactly analogous to
`PlaceholderPolicyEngine.score` echoing back the `action` it was given —
see `app/policies/engine.py`) but never computes anything from the
individual results' scores, `passed` values, or confidences. These exist
solely to provide working, testable implementations of the
`AggregationStrategy` interface until real aggregation algorithms are
implemented.
"""

from abc import ABC, abstractmethod

from app.critics.models import AggregatedCriticResult, CriticResult


class AggregationStrategy(ABC):
    """Abstract interface for a strategy that combines multiple `CriticResult`s
    into a single `AggregatedCriticResult`.

    This defines only the contract; no aggregation algorithm is
    implemented here. Concrete subclasses must set `strategy_name` and
    implement `aggregate`.
    """

    strategy_name: str = "AggregationStrategy"

    @abstractmethod
    def aggregate(self, results: list[CriticResult]) -> AggregatedCriticResult:
        """Combine `results` into a single `AggregatedCriticResult`.

        Concrete subclasses must implement the actual aggregation
        algorithm; no aggregation logic is implemented here.

        Args:
            results: The individual critic results to combine.

        Returns:
            An `AggregatedCriticResult` describing the combined outcome.
        """
        raise NotImplementedError

    def _placeholder_result(self, results: list[CriticResult]) -> AggregatedCriticResult:
        """Build a fixed, neutral `AggregatedCriticResult` for this strategy.

        Shared by the placeholder subclasses below so each `aggregate`
        need not duplicate `AggregatedCriticResult` construction. This
        performs no real aggregation: `aggregated_score`,
        `aggregated_passed`, and `confidence` are always the same fixed
        values regardless of `results`. Only `contributing_critics` and
        `individual_results` are derived from `results`, as structural
        bookkeeping rather than an aggregation computation.

        Args:
            results: The individual critic results being "aggregated".

        Returns:
            An `AggregatedCriticResult` with `aggregated_score=0.0`,
            `confidence=0.0`, `aggregated_passed=None`, and a fixed
            placeholder `rationale`.
        """
        return AggregatedCriticResult(
            strategy_name=self.strategy_name,
            aggregated_score=0.0,
            aggregated_passed=None,
            confidence=0.0,
            contributing_critics=[result.critic_name for result in results],
            individual_results=list(results),
            rationale=f"{self.strategy_name} is a placeholder: no aggregation algorithm implemented.",
            metadata={"strategy_class": type(self).__name__, "result_count": len(results)},
        )


class MajorityVoteStrategy(AggregationStrategy):
    """Placeholder strategy for majority-vote aggregation. No vote counting is implemented."""

    strategy_name = "MajorityVoteStrategy"

    def aggregate(self, results: list[CriticResult]) -> AggregatedCriticResult:
        """Return a fixed placeholder `AggregatedCriticResult`."""
        return self._placeholder_result(results)


class WeightedAverageStrategy(AggregationStrategy):
    """Placeholder strategy for weighted-average aggregation. No averaging is implemented."""

    strategy_name = "WeightedAverageStrategy"

    def aggregate(self, results: list[CriticResult]) -> AggregatedCriticResult:
        """Return a fixed placeholder `AggregatedCriticResult`."""
        return self._placeholder_result(results)


class MaxScoreStrategy(AggregationStrategy):
    """Placeholder strategy for max-score aggregation. No max-selection is implemented."""

    strategy_name = "MaxScoreStrategy"

    def aggregate(self, results: list[CriticResult]) -> AggregatedCriticResult:
        """Return a fixed placeholder `AggregatedCriticResult`."""
        return self._placeholder_result(results)


class PolicyWeightedStrategy(AggregationStrategy):
    """Placeholder strategy for policy-weighted aggregation. No weighting is implemented."""

    strategy_name = "PolicyWeightedStrategy"

    def aggregate(self, results: list[CriticResult]) -> AggregatedCriticResult:
        """Return a fixed placeholder `AggregatedCriticResult`."""
        return self._placeholder_result(results)
