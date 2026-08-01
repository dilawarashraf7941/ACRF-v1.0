"""Unit tests for the policy engine data structures: `PolicyState`,
`CriticAction`, and `PolicyScore` (see `app/policies/engine.py`).

Per scope, this file tests only these data structures — not the
`PolicyEngine` interface or `PlaceholderPolicyEngine`.
"""

import pytest
from pydantic import ValidationError

from app.policies import CriticAction, CriticActionType, PolicyScore, PolicyState


# --- PolicyState ---


def test_policy_state_requires_session_and_task_id() -> None:
    with pytest.raises(ValidationError):
        PolicyState()  # type: ignore[call-arg]


def test_policy_state_applies_defaults() -> None:
    state = PolicyState(session_id="s1", task_id="t1")

    assert state.task_type is None
    assert state.iteration_count == 0
    assert state.max_iterations == 10
    assert state.critic_scores == {}
    assert state.context == {}


def test_policy_state_accepts_explicit_values() -> None:
    state = PolicyState(
        session_id="s1",
        task_id="t1",
        task_type="code",
        iteration_count=2,
        max_iterations=5,
        critic_scores={"LogicCritic": 0.8},
        context={"risk_level": "medium"},
    )

    assert state.task_type == "code"
    assert state.iteration_count == 2
    assert state.max_iterations == 5
    assert state.critic_scores == {"LogicCritic": 0.8}
    assert state.context == {"risk_level": "medium"}


def test_policy_state_rejects_negative_iteration_count() -> None:
    with pytest.raises(ValidationError):
        PolicyState(session_id="s1", task_id="t1", iteration_count=-1)


def test_policy_state_rejects_negative_max_iterations() -> None:
    with pytest.raises(ValidationError):
        PolicyState(session_id="s1", task_id="t1", max_iterations=-1)


def test_policy_state_allows_extra_fields() -> None:
    state = PolicyState(session_id="s1", task_id="t1", custom_signal="anomaly_detected")

    assert state.custom_signal == "anomaly_detected"  # type: ignore[attr-defined]


# --- CriticAction ---


def test_critic_action_requires_action_type() -> None:
    with pytest.raises(ValidationError):
        CriticAction()  # type: ignore[call-arg]


def test_critic_action_applies_defaults() -> None:
    action = CriticAction(action_type=CriticActionType.SKIP_CRITIC)

    assert action.critic_id is None
    assert action.rationale is None
    assert action.metadata == {}


def test_critic_action_accepts_explicit_values() -> None:
    action = CriticAction(
        action_type=CriticActionType.INVOKE_CRITIC,
        critic_id="CodeCritic",
        rationale="Task type is code.",
        metadata={"priority": 1},
    )

    assert action.action_type == CriticActionType.INVOKE_CRITIC
    assert action.critic_id == "CodeCritic"
    assert action.rationale == "Task type is code."
    assert action.metadata == {"priority": 1}


def test_critic_action_rejects_invalid_action_type() -> None:
    with pytest.raises(ValidationError):
        CriticAction(action_type="not_a_real_action_type")


@pytest.mark.parametrize(
    "action_type",
    [
        CriticActionType.INVOKE_CRITIC,
        CriticActionType.SKIP_CRITIC,
        CriticActionType.INVOKE_META_CRITIC,
        CriticActionType.REQUEST_SELF_CORRECTION,
        CriticActionType.FINALIZE,
    ],
)
def test_critic_action_type_enum_members_are_all_constructible(action_type: CriticActionType) -> None:
    action = CriticAction(action_type=action_type)

    assert action.action_type == action_type


def test_critic_action_allows_extra_fields() -> None:
    action = CriticAction(action_type=CriticActionType.FINALIZE, custom_flag=True)

    assert action.custom_flag is True  # type: ignore[attr-defined]


# --- PolicyScore ---


def test_policy_score_requires_action_and_score() -> None:
    with pytest.raises(ValidationError):
        PolicyScore()  # type: ignore[call-arg]


def test_policy_score_applies_defaults() -> None:
    action = CriticAction(action_type=CriticActionType.INVOKE_CRITIC, critic_id="LogicCritic")

    score = PolicyScore(action=action, score=0.5)

    assert score.rationale is None
    assert score.metadata == {}


def test_policy_score_accepts_explicit_values() -> None:
    action = CriticAction(action_type=CriticActionType.INVOKE_META_CRITIC)

    score = PolicyScore(
        action=action,
        score=0.75,
        rationale="High disagreement between critics.",
        metadata={"source": "test"},
    )

    assert score.action == action
    assert score.score == 0.75
    assert score.rationale == "High disagreement between critics."
    assert score.metadata == {"source": "test"}


def test_policy_score_accepts_negative_and_unbounded_scores() -> None:
    # The scoring scale is intentionally left undefined by this schema.
    action = CriticAction(action_type=CriticActionType.SKIP_CRITIC)

    negative = PolicyScore(action=action, score=-5.0)
    large = PolicyScore(action=action, score=1000.0)

    assert negative.score == -5.0
    assert large.score == 1000.0


def test_policy_score_rejects_missing_action() -> None:
    with pytest.raises(ValidationError):
        PolicyScore(score=1.0)  # type: ignore[call-arg]


def test_policy_score_nested_action_round_trips_via_dict() -> None:
    action = CriticAction(action_type=CriticActionType.REQUEST_SELF_CORRECTION, critic_id="LogicCritic")
    score = PolicyScore(action=action, score=0.2)

    dumped = score.model_dump()

    assert dumped["action"]["action_type"] == "request_self_correction"
    assert dumped["action"]["critic_id"] == "LogicCritic"
    assert dumped["score"] == 0.2


def test_policy_score_allows_extra_fields() -> None:
    action = CriticAction(action_type=CriticActionType.FINALIZE)

    score = PolicyScore(action=action, score=1.0, confidence_interval=[0.9, 1.0])

    assert score.confidence_interval == [0.9, 1.0]  # type: ignore[attr-defined]
