"""Unit tests for `ContextVector` (app/context/models.py)."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.context import ContextVector


def _make_context(**overrides: object) -> ContextVector:
    defaults: dict[str, object] = {
        "context_id": "ctx-1",
        "timestamp": datetime.now(timezone.utc),
    }
    defaults.update(overrides)
    return ContextVector(**defaults)  # type: ignore[arg-type]


def test_requires_context_id_and_timestamp() -> None:
    with pytest.raises(ValidationError):
        ContextVector()  # type: ignore[call-arg]


def test_applies_defaults() -> None:
    context = _make_context()

    assert context.source_execution_id is None
    assert context.features == {}
    assert context.feature_order == []
    assert context.normalized is False
    assert context.normalization_strategy is None
    assert context.metadata == {}


def test_accepts_all_fields_explicitly() -> None:
    context = _make_context(
        source_execution_id="exp-1",
        features={"iteration_count": 2.0},
        feature_order=["iteration_count"],
        normalized=True,
        normalization_strategy="fixed_bounds_min_max",
        metadata={"source": "AgentState"},
    )

    assert context.source_execution_id == "exp-1"
    assert context.features == {"iteration_count": 2.0}
    assert context.feature_order == ["iteration_count"]
    assert context.normalized is True
    assert context.normalization_strategy == "fixed_bounds_min_max"
    assert context.metadata == {"source": "AgentState"}


def test_is_frozen() -> None:
    context = _make_context()

    with pytest.raises(ValidationError):
        context.context_id = "changed"  # type: ignore[misc]


def test_allows_extra_fields() -> None:
    context = _make_context(custom_field="value")

    assert context.custom_field == "value"  # type: ignore[attr-defined]


def test_round_trips_via_model_dump() -> None:
    context = _make_context(features={"a": 1.0, "b": 2.0}, feature_order=["a", "b"])

    dumped = context.model_dump(mode="json")
    reconstructed = ContextVector(**dumped)

    assert reconstructed == context
