"""Unit tests for `PolicyDecision` (`app/policy/models.py`)."""

import pytest
from pydantic import ValidationError

from app.policy.models import PolicyDecision


def _make_decision(**overrides: object) -> PolicyDecision:
    defaults: dict[str, object] = {
        "selected_critics": ["LogicCritic"],
        "scores": {"LogicCritic": 0.5, "CodeCritic": 0.2},
        "ranking": [
            {"critic_name": "LogicCritic", "score": 0.5, "rank": 1},
            {"critic_name": "CodeCritic", "score": 0.2, "rank": 2},
        ],
        "policy_name": "TestPolicy",
        "policy_version": "1.0.0",
        "confidence": 0.5,
    }
    defaults.update(overrides)
    return PolicyDecision(**defaults)


def test_constructs_with_required_fields_only() -> None:
    decision = PolicyDecision(
        selected_critics=["LogicCritic"],
        policy_name="TestPolicy",
        policy_version="1.0.0",
        confidence=0.5,
    )

    assert decision.selected_critics == ["LogicCritic"]
    assert decision.scores == {}
    assert decision.ranking == []
    assert decision.metadata == {}


def test_missing_selected_critics_raises() -> None:
    with pytest.raises(ValidationError):
        PolicyDecision(policy_name="TestPolicy", policy_version="1.0.0", confidence=0.5)


def test_missing_policy_name_raises() -> None:
    with pytest.raises(ValidationError):
        PolicyDecision(selected_critics=[], policy_version="1.0.0", confidence=0.5)


def test_missing_confidence_raises() -> None:
    with pytest.raises(ValidationError):
        PolicyDecision(selected_critics=[], policy_name="TestPolicy", policy_version="1.0.0")


@pytest.mark.parametrize("confidence", [-0.01, 1.01, -1.0, 2.0])
def test_confidence_out_of_bounds_raises(confidence: float) -> None:
    with pytest.raises(ValidationError):
        _make_decision(confidence=confidence)


@pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
def test_confidence_boundary_values_are_valid(confidence: float) -> None:
    decision = _make_decision(confidence=confidence)
    assert decision.confidence == confidence


def test_is_frozen() -> None:
    decision = _make_decision()

    with pytest.raises(ValidationError):
        decision.confidence = 0.9  # type: ignore[misc]


def test_allows_extra_fields() -> None:
    decision = PolicyDecision(
        selected_critics=[],
        policy_name="TestPolicy",
        policy_version="1.0.0",
        confidence=0.0,
        extra_signal="anything",  # type: ignore[call-arg]
    )

    assert decision.model_dump()["extra_signal"] == "anything"


def test_metadata_holds_arbitrary_diagnostics() -> None:
    decision = _make_decision(metadata={"selection_strategy": "top_1", "context_id": "abc"})

    assert decision.metadata == {"selection_strategy": "top_1", "context_id": "abc"}
