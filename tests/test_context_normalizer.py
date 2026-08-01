"""Unit tests for `ContextNormalizer` (app/context/normalizer.py)."""

from datetime import datetime, timezone

import pytest

from app.context import ContextNormalizer, ContextVector
from app.context.normalizer import FEATURE_BOUNDS, NORMALIZATION_STRATEGY_NAME


def _make_context(**overrides: object) -> ContextVector:
    defaults: dict[str, object] = {
        "context_id": "ctx-1",
        "timestamp": datetime.now(timezone.utc),
    }
    defaults.update(overrides)
    return ContextVector(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def normalizer() -> ContextNormalizer:
    return ContextNormalizer()


# --- Basic contract ---


def test_normalize_returns_new_context_vector(normalizer: ContextNormalizer) -> None:
    context = _make_context(features={"iteration_count": 5.0}, feature_order=["iteration_count"])

    normalized = normalizer.normalize(context)

    assert isinstance(normalized, ContextVector)
    assert normalized is not context


def test_normalize_sets_normalized_flag_and_strategy(normalizer: ContextNormalizer) -> None:
    context = _make_context(features={"iteration_count": 5.0})

    normalized = normalizer.normalize(context)

    assert normalized.normalized is True
    assert normalized.normalization_strategy == NORMALIZATION_STRATEGY_NAME


def test_normalize_does_not_mutate_original(normalizer: ContextNormalizer) -> None:
    context = _make_context(features={"iteration_count": 5.0})

    normalizer.normalize(context)

    assert context.normalized is False
    assert context.features == {"iteration_count": 5.0}


def test_normalize_preserves_feature_order(normalizer: ContextNormalizer) -> None:
    context = _make_context(
        features={"iteration_count": 5.0, "max_iterations": 10.0},
        feature_order=["iteration_count", "max_iterations"],
    )

    normalized = normalizer.normalize(context)

    assert normalized.feature_order == ["iteration_count", "max_iterations"]


def test_normalize_preserves_other_fields(normalizer: ContextNormalizer) -> None:
    context = _make_context(
        context_id="ctx-abc", source_execution_id="exp-1", metadata={"source": "AgentState"}
    )

    normalized = normalizer.normalize(context)

    assert normalized.context_id == "ctx-abc"
    assert normalized.source_execution_id == "exp-1"
    assert normalized.metadata == {"source": "AgentState"}


# --- Scaling correctness ---


def test_scales_value_at_lower_bound_to_zero(normalizer: ContextNormalizer) -> None:
    context = _make_context(features={"iteration_count": 0.0})

    normalized = normalizer.normalize(context)

    assert normalized.features["iteration_count"] == 0.0


def test_scales_value_at_upper_bound_to_one(normalizer: ContextNormalizer) -> None:
    low, high = FEATURE_BOUNDS["iteration_count"]
    context = _make_context(features={"iteration_count": high})

    normalized = normalizer.normalize(context)

    assert normalized.features["iteration_count"] == 1.0


def test_scales_midpoint_to_half(normalizer: ContextNormalizer) -> None:
    low, high = FEATURE_BOUNDS["iteration_count"]
    midpoint = (low + high) / 2

    context = _make_context(features={"iteration_count": midpoint})
    normalized = normalizer.normalize(context)

    assert normalized.features["iteration_count"] == pytest.approx(0.5)


def test_clamps_value_below_lower_bound(normalizer: ContextNormalizer) -> None:
    context = _make_context(features={"iteration_count": -100.0})

    normalized = normalizer.normalize(context)

    assert normalized.features["iteration_count"] == 0.0


def test_clamps_value_above_upper_bound(normalizer: ContextNormalizer) -> None:
    context = _make_context(features={"iteration_count": 1_000_000.0})

    normalized = normalizer.normalize(context)

    assert normalized.features["iteration_count"] == 1.0


def test_all_known_features_produce_values_within_zero_one(normalizer: ContextNormalizer) -> None:
    # Use each feature's own upper bound (and beyond) to exercise the full table.
    features = {name: high * 2 for name, (_, high) in FEATURE_BOUNDS.items()}
    context = _make_context(features=features, feature_order=list(features.keys()))

    normalized = normalizer.normalize(context)

    assert all(0.0 <= value <= 1.0 for value in normalized.features.values())


# --- Graceful degradation ---


def test_unknown_feature_name_passes_through_unchanged(normalizer: ContextNormalizer) -> None:
    context = _make_context(features={"some_future_feature": 42.0})

    normalized = normalizer.normalize(context)

    assert normalized.features["some_future_feature"] == 42.0


def test_empty_features_produce_empty_normalized_features(normalizer: ContextNormalizer) -> None:
    context = _make_context(features={})

    normalized = normalizer.normalize(context)

    assert normalized.features == {}


def test_degenerate_bounds_pass_through_unchanged() -> None:
    normalizer = ContextNormalizer(bounds={"weird_feature": (5.0, 5.0)})
    context = _make_context(features={"weird_feature": 5.0})

    normalized = normalizer.normalize(context)

    assert normalized.features["weird_feature"] == 5.0


# --- Dependency injection ---


def test_custom_bounds_table_is_used_instead_of_default() -> None:
    custom_normalizer = ContextNormalizer(bounds={"iteration_count": (0.0, 2.0)})
    context = _make_context(features={"iteration_count": 1.0})

    normalized = custom_normalizer.normalize(context)

    assert normalized.features["iteration_count"] == 0.5


def test_default_bounds_used_when_none_injected(normalizer: ContextNormalizer) -> None:
    context = _make_context(features={"iteration_count": FEATURE_BOUNDS["iteration_count"][1]})

    normalized = normalizer.normalize(context)

    assert normalized.features["iteration_count"] == 1.0


# --- Determinism ---


def test_normalize_is_deterministic(normalizer: ContextNormalizer) -> None:
    context = _make_context(features={"iteration_count": 4.0, "max_iterations": 8.0})

    normalized_a = normalizer.normalize(context)
    normalized_b = normalizer.normalize(context)

    assert normalized_a.features == normalized_b.features


def test_feature_bounds_table_has_no_degenerate_ranges() -> None:
    """Sanity check on the shipped default table itself."""
    assert all(high > low for low, high in FEATURE_BOUNDS.values())
