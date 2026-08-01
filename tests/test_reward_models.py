"""Unit tests for `RewardSignal` (app/reward/models.py)."""

import pytest
from pydantic import ValidationError

from app.reward import RewardSignal


def _make_signal(**overrides: object) -> RewardSignal:
    defaults: dict[str, object] = {
        "reward": 0.5,
        "quality_reward": 0.5,
        "efficiency_penalty": 0.0,
        "cost_penalty": 0.0,
        "latency_penalty": 0.0,
        "correction_penalty": 0.0,
        "completion_bonus": 0.0,
        "confidence": 1.0,
        "strategy": "WeightedRewardStrategy",
        "explanation": "test",
    }
    defaults.update(overrides)
    return RewardSignal(**defaults)  # type: ignore[arg-type]


def test_requires_core_fields() -> None:
    with pytest.raises(ValidationError):
        RewardSignal()  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "field",
    [
        "reward",
        "quality_reward",
        "efficiency_penalty",
        "cost_penalty",
        "latency_penalty",
        "correction_penalty",
        "completion_bonus",
        "confidence",
        "strategy",
        "explanation",
    ],
)
def test_required_fields_are_enforced(field: str) -> None:
    kwargs = {
        "reward": 0.5,
        "quality_reward": 0.5,
        "efficiency_penalty": 0.0,
        "cost_penalty": 0.0,
        "latency_penalty": 0.0,
        "correction_penalty": 0.0,
        "completion_bonus": 0.0,
        "confidence": 1.0,
        "strategy": "WeightedRewardStrategy",
        "explanation": "test",
    }
    del kwargs[field]

    with pytest.raises(ValidationError):
        RewardSignal(**kwargs)  # type: ignore[arg-type]


def test_metadata_defaults_to_empty_dict() -> None:
    signal = _make_signal()

    assert signal.metadata == {}


def test_accepts_all_fields_explicitly() -> None:
    signal = _make_signal(
        reward=0.75,
        quality_reward=0.9,
        efficiency_penalty=0.1,
        cost_penalty=0.05,
        latency_penalty=0.05,
        correction_penalty=0.1,
        completion_bonus=0.2,
        confidence=0.8,
        metadata={"experience_id": "abc"},
    )

    assert signal.reward == 0.75
    assert signal.quality_reward == 0.9
    assert signal.efficiency_penalty == 0.1
    assert signal.cost_penalty == 0.05
    assert signal.latency_penalty == 0.05
    assert signal.correction_penalty == 0.1
    assert signal.completion_bonus == 0.2
    assert signal.confidence == 0.8
    assert signal.metadata == {"experience_id": "abc"}


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_confidence_must_be_within_bounds(confidence: float) -> None:
    with pytest.raises(ValidationError):
        _make_signal(confidence=confidence)


def test_confidence_boundaries_are_valid() -> None:
    assert _make_signal(confidence=0.0).confidence == 0.0
    assert _make_signal(confidence=1.0).confidence == 1.0


def test_reward_and_penalties_are_unbounded_floats() -> None:
    """Negative rewards and arbitrarily large penalties must be representable."""
    signal = _make_signal(reward=-5.0, cost_penalty=100.0)

    assert signal.reward == -5.0
    assert signal.cost_penalty == 100.0


def test_completion_bonus_may_be_negative() -> None:
    signal = _make_signal(completion_bonus=-0.3)

    assert signal.completion_bonus == -0.3


def test_is_frozen() -> None:
    signal = _make_signal()

    with pytest.raises(ValidationError):
        signal.reward = 99.0  # type: ignore[misc]


def test_allows_extra_fields() -> None:
    signal = _make_signal(custom_field="value")

    assert signal.custom_field == "value"  # type: ignore[attr-defined]


def test_round_trips_via_model_dump() -> None:
    signal = _make_signal(metadata={"weights": {"quality_reward_weight": 1.0}})

    dumped = signal.model_dump(mode="json")
    reconstructed = RewardSignal(**dumped)

    assert reconstructed == signal
