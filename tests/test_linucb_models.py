"""Unit tests for `LinUCBPrediction` and `LinUCBSelection` (`app/policy/linucb/models.py`)."""

import pytest
from pydantic import ValidationError

from app.policy.linucb.models import LinUCBPrediction, LinUCBSelection


def _make_prediction(**overrides: object) -> LinUCBPrediction:
    defaults: dict[str, object] = {
        "arm_id": "LogicCritic",
        "expected_reward": 0.5,
        "confidence_bonus": 0.3,
        "upper_confidence_bound": 0.8,
        "context_id": "ctx-1",
    }
    defaults.update(overrides)
    return LinUCBPrediction(**defaults)


def test_prediction_constructs_with_all_fields() -> None:
    prediction = _make_prediction()
    assert prediction.arm_id == "LogicCritic"
    assert prediction.upper_confidence_bound == 0.8


def test_prediction_confidence_bonus_must_be_non_negative() -> None:
    with pytest.raises(ValidationError):
        _make_prediction(confidence_bonus=-0.01)


def test_prediction_is_frozen() -> None:
    prediction = _make_prediction()
    with pytest.raises(ValidationError):
        prediction.expected_reward = 1.0  # type: ignore[misc]


def test_prediction_missing_required_field_raises() -> None:
    with pytest.raises(ValidationError):
        LinUCBPrediction(arm_id="LogicCritic")


def test_selection_constructs_with_predictions_map() -> None:
    predictions = {
        "LogicCritic": _make_prediction(),
        "CodeCritic": _make_prediction(arm_id="CodeCritic"),
    }

    selection = LinUCBSelection(
        selected_action="LogicCritic",
        predictions=predictions,
        alpha=1.0,
        context_id="ctx-1",
    )

    assert selection.selected_action == "LogicCritic"
    assert set(selection.predictions.keys()) == {"LogicCritic", "CodeCritic"}


def test_selection_alpha_must_be_non_negative() -> None:
    with pytest.raises(ValidationError):
        LinUCBSelection(
            selected_action="LogicCritic",
            predictions={},
            alpha=-1.0,
            context_id="ctx-1",
        )


def test_selection_is_frozen() -> None:
    selection = LinUCBSelection(
        selected_action="LogicCritic", predictions={}, alpha=1.0, context_id="ctx-1"
    )
    with pytest.raises(ValidationError):
        selection.selected_action = "CodeCritic"  # type: ignore[misc]
