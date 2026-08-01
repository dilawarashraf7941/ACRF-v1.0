"""Deterministic critic selection strategies over a `CriticRanking`.

Contains no scoring or ranking logic of its own (see
`app/policy_engine/scorer.py` and `app/policy_engine/ranking.py`) and no
learning — every strategy here is a fixed, deterministic rule applied to
an already-computed `CriticRanking`.
"""

from enum import Enum

from app.policy_engine.ranking import CriticRanking


class SelectionStrategy(str, Enum):
    """The selection strategies `CriticSelector` supports."""

    TOP_1 = "top_1"
    TOP_K = "top_k"
    THRESHOLD = "threshold"


class CriticSelector:
    """Deterministically selects critics from a `CriticRanking`."""

    def select_top_1(self, ranking: CriticRanking) -> list[str]:
        """Select the single highest-ranked critic.

        Args:
            ranking: The ranking to select from.

        Returns:
            A single-element list with the top critic's identifier, or an
            empty list if `ranking` has no candidates.
        """
        return [entry.critic_name for entry in ranking.top(1)]

    def select_top_k(self, ranking: CriticRanking, k: int) -> list[str]:
        """Select the `k` highest-ranked critics.

        Args:
            ranking: The ranking to select from.
            k: The number of critics to select. Must be non-negative.

        Returns:
            Up to `k` critic identifiers, highest-ranked first.

        Raises:
            ValueError: If `k` is negative.
        """
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        return [entry.critic_name for entry in ranking.top(k)]

    def select_by_threshold(self, ranking: CriticRanking, threshold: float) -> list[str]:
        """Select every critic whose score is at or above `threshold`.

        Args:
            ranking: The ranking to select from.
            threshold: The minimum (inclusive) score a critic must have to be selected.

        Returns:
            Critic identifiers meeting the threshold, highest-ranked first.
            May be empty if no critic meets the threshold.
        """
        return [entry.critic_name for entry in ranking.ranked_critics if entry.score >= threshold]

    def select(
        self,
        ranking: CriticRanking,
        strategy: SelectionStrategy,
        *,
        k: int | None = None,
        threshold: float | None = None,
    ) -> list[str]:
        """Select critics from `ranking` using the named `strategy`.

        A single dispatch entry point over `select_top_1` /
        `select_top_k` / `select_by_threshold`, for callers that pick a
        strategy dynamically (e.g. `policy_engine_node`).

        Args:
            ranking: The ranking to select from.
            strategy: Which selection strategy to apply.
            k: Required, and only used, when `strategy` is `TOP_K`.
            threshold: Required, and only used, when `strategy` is `THRESHOLD`.

        Returns:
            The critic identifiers selected by the chosen strategy.

        Raises:
            ValueError: If `strategy` is `TOP_K`/`THRESHOLD` and the
                corresponding required argument is missing, or if
                `strategy` is not a recognized `SelectionStrategy`.
        """
        if strategy == SelectionStrategy.TOP_1:
            return self.select_top_1(ranking)
        if strategy == SelectionStrategy.TOP_K:
            if k is None:
                raise ValueError("k is required for the TOP_K selection strategy")
            return self.select_top_k(ranking, k)
        if strategy == SelectionStrategy.THRESHOLD:
            if threshold is None:
                raise ValueError("threshold is required for the THRESHOLD selection strategy")
            return self.select_by_threshold(ranking, threshold)
        raise ValueError(f"Unknown selection strategy: {strategy!r}")
