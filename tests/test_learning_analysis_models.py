"""Unit tests for `LearningCurve` (`app/evaluation/learning_analysis/models.py`)."""

import pytest
from pydantic import ValidationError

from app.evaluation.learning_analysis.models import LearningCurve


def _make_curve(**overrides: object) -> LearningCurve:
    defaults: dict[str, object] = {
        "reward_per_step": [1.0, 2.0],
        "cumulative_reward": [1.0, 3.0],
        "instantaneous_regret": [1.0, 0.0],
        "cumulative_regret": [1.0, 1.0],
        "average_reward": 1.5,
        "moving_average_reward": [1.0, 1.5],
    }
    defaults.update(overrides)
    return LearningCurve(**defaults)


def test_constructs_with_defaults() -> None:
    curve = LearningCurve()
    assert curve.reward_per_step == []
    assert curve.average_reward == 0.0
    assert curve.metadata == {}


def test_constructs_with_explicit_values() -> None:
    curve = _make_curve()
    assert curve.reward_per_step == [1.0, 2.0]
    assert curve.cumulative_reward == [1.0, 3.0]


def test_is_frozen() -> None:
    curve = _make_curve()
    with pytest.raises(ValidationError):
        curve.average_reward = 5.0  # type: ignore[misc]


def test_rejects_mismatched_series_lengths() -> None:
    with pytest.raises(ValidationError, match="same length"):
        _make_curve(reward_per_step=[1.0, 2.0, 3.0])


def test_rejects_mismatched_moving_average_length() -> None:
    with pytest.raises(ValidationError, match="same length"):
        _make_curve(moving_average_reward=[1.0])


def test_allows_extra_fields() -> None:
    curve = LearningCurve(extra_signal="anything")  # type: ignore[call-arg]
    assert curve.model_dump()["extra_signal"] == "anything"


def test_holds_metadata() -> None:
    curve = _make_curve(metadata={"convergence_point": 1, "learning_rate_estimate": 0.5})
    assert curve.metadata["convergence_point"] == 1
