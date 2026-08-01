"""`ContextNormalizer`: deterministically rescales a `ContextVector`'s features.

No reinforcement learning, no contextual bandits, no policy optimization,
and no learning of any kind — every bound below is a fixed constant
chosen at implementation time. Critically, bounds are **never fitted from
a batch of `ContextVector`s** (that would be learning a distribution's
statistics from data); this normalizer only ever applies a fixed,
pre-declared min-max table, matching the same "fixed, hand-authored
heuristic" convention used throughout `app/policy_engine`,
`app/correction_policy`, and `app/reward`.
"""

from app.context.models import ContextVector

NORMALIZATION_STRATEGY_NAME = "fixed_bounds_min_max"
"""Identifier recorded on a normalized `ContextVector.normalization_strategy`."""

FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "iteration_count": (0.0, 20.0),
    "max_iterations": (1.0, 20.0),
    "iteration_ratio": (0.0, 1.0),
    "error_feature_count": (0.0, 20.0),
    "worker_output_count": (0.0, 20.0),
    "critic_score_count": (0.0, 10.0),
    "selected_critics_count": (0.0, 10.0),
    "retrieved_memories_count": (0.0, 50.0),
    "correction_history_count": (0.0, 20.0),
    "aggregated_quality_score": (0.0, 1.0),
    "has_aggregated_quality_score": (0.0, 1.0),
    "safety_status_code": (0.0, 3.0),
    "execution_status_code": (0.0, 5.0),
    "is_code_task": (0.0, 1.0),
    "has_task_type": (0.0, 1.0),
    "average_critic_score": (0.0, 1.0),
    "max_critic_score": (0.0, 1.0),
    "min_critic_score": (0.0, 1.0),
    "uncertainty": (0.0, 1.0),
    "risk": (0.0, 1.0),
    "task_complexity": (0.0, 1.0),
    "memory_relevance": (0.0, 1.0),
    "requires_self_correction": (0.0, 1.0),
    "requires_meta_critic": (0.0, 1.0),
    "is_code_output": (0.0, 1.0),
    "iteration_pressure": (0.0, 1.0),
    "attempt_pressure": (0.0, 1.0),
}
"""Fixed `(min, max)` bounds for every feature `ContextEncoder` produces (see
`app/context/encoder.py`). A feature not present here is passed through
unchanged by the default `ContextNormalizer` (graceful degradation for a
future encoder version's new feature names)."""


class ContextNormalizer:
    """Deterministically min-max-scales a `ContextVector`'s `features` into `[0.0, 1.0]`.

    The bounds table is injected via the constructor (dependency
    injection), defaulting to the fixed `FEATURE_BOUNDS` above.
    """

    def __init__(self, bounds: dict[str, tuple[float, float]] | None = None) -> None:
        """Create a normalizer, optionally wired to a specific bounds table.

        Args:
            bounds: The `{feature_name: (min, max)}` table to use.
                Defaults to `FEATURE_BOUNDS` when not provided.
        """
        self._bounds = bounds if bounds is not None else dict(FEATURE_BOUNDS)

    def normalize(self, context: ContextVector) -> ContextVector:
        """Return a new `ContextVector` with every feature scaled into `[0.0, 1.0]`.

        `context` itself is never mutated (it is frozen); this returns a
        new instance with `normalized=True` and
        `normalization_strategy` set.

        Args:
            context: The `ContextVector` to rescale.

        Returns:
            A new `ContextVector` with the same `feature_order` and every
            value in `features` clamped to `[0.0, 1.0]`.
        """
        normalized_features = {
            name: self._normalize_value(name, value) for name, value in context.features.items()
        }
        return context.model_copy(
            update={
                "features": normalized_features,
                "normalized": True,
                "normalization_strategy": NORMALIZATION_STRATEGY_NAME,
            }
        )

    def _normalize_value(self, name: str, value: float) -> float:
        """Min-max-scale a single named value using this normalizer's bounds table.

        Args:
            name: The feature name, used to look up its `(min, max)` bounds.
            value: The raw value to scale.

        Returns:
            `clamp((value - min) / (max - min), 0.0, 1.0)` if `name` has
            known bounds and `max > min`; otherwise `value` unchanged
            (graceful degradation for an unrecognized feature or a
            degenerate `[min, max]` range).
        """
        bounds = self._bounds.get(name)
        if bounds is None:
            return value
        low, high = bounds
        if high <= low:
            return value
        scaled = (value - low) / (high - low)
        return max(0.0, min(1.0, scaled))
