"""Deterministic ranking of critics by score.

Contains no scoring logic of its own (see `app/policy_engine/scorer.py`)
and no learning — `CriticRanking` only orders an already-computed
`dict[str, float]`.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RankedCritic:
    """A single critic's position within a `CriticRanking`."""

    critic_name: str
    """Identifier of the critic."""

    score: float
    """The score this critic was ranked by."""

    rank: int
    """1-based rank; `1` is the highest-scoring critic."""


class CriticRanking:
    """Deterministically orders critics from highest score to lowest.

    Ties are broken by critic name in ascending alphabetical order, so
    two calls with the same `scores` mapping — regardless of key
    insertion order, since plain `dict`s do not guarantee a meaningful
    order here — always produce an identical ranking.
    """

    def __init__(self, scores: dict[str, float]) -> None:
        """Build a ranking from `scores`.

        Args:
            scores: A mapping of critic identifier to numeric score.
        """
        self._scores = dict(scores)
        self._ranked = self._build_ranking(self._scores)

    @staticmethod
    def _build_ranking(scores: dict[str, float]) -> list[RankedCritic]:
        """Sort `scores` into `RankedCritic` entries: highest score first, ties broken by name."""
        ordered_names = sorted(scores.keys(), key=lambda name: (-scores[name], name))
        return [
            RankedCritic(critic_name=name, score=scores[name], rank=position)
            for position, name in enumerate(ordered_names, start=1)
        ]

    @property
    def ranked_critics(self) -> list[RankedCritic]:
        """The full ranking, highest score first."""
        return list(self._ranked)

    def top(self, n: int = 1) -> list[RankedCritic]:
        """Return the top `n` ranked critics.

        Args:
            n: Number of top-ranked critics to return. Must be non-negative.

        Returns:
            Up to `n` `RankedCritic` entries, highest score first. If `n`
            exceeds the number of ranked critics, all of them are returned.

        Raises:
            ValueError: If `n` is negative.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        return self._ranked[:n]

    def critic_names(self) -> list[str]:
        """Return just the critic identifiers, highest score first."""
        return [entry.critic_name for entry in self._ranked]

    def score_for(self, critic_name: str) -> float:
        """Return the raw score for `critic_name`.

        Args:
            critic_name: The critic identifier to look up.

        Returns:
            The score originally passed in for `critic_name`.

        Raises:
            KeyError: If `critic_name` was not part of the scored candidates.
        """
        return self._scores[critic_name]

    def as_list_of_dicts(self) -> list[dict[str, str | float | int]]:
        """Return the ranking as plain dicts, e.g. for diagnostics or serialization."""
        return [
            {"critic_name": entry.critic_name, "score": entry.score, "rank": entry.rank}
            for entry in self._ranked
        ]
