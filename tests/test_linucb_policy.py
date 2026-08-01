"""Unit tests for `LinUCBPolicy` (`app/policy/linucb/policy.py`).

Covers: best arm selection, multiple independent arms, deterministic
behavior, and dimension consistency across the policy's lazily-created
arms.
"""

import numpy as np
import pytest

from app.context import ContextVector
from app.policy.linucb.models import LinUCBSelection
from app.policy.linucb.policy import LinUCBPolicy

_ACTIONS = ["LogicCritic", "CodeCritic", "FactCritic", "MetaCritic"]


def _make_context(features: dict[str, float], context_id: str = "ctx-1") -> ContextVector:
    return ContextVector(
        context_id=context_id,
        features=features,
        feature_order=list(features.keys()),
        timestamp="2024-01-01T00:00:00Z",
    )


def test_negative_alpha_raises() -> None:
    with pytest.raises(ValueError):
        LinUCBPolicy(alpha=-1.0)


@pytest.mark.parametrize("regularization", [0.0, -1.0])
def test_non_positive_regularization_raises(regularization: float) -> None:
    with pytest.raises(ValueError):
        LinUCBPolicy(regularization=regularization)


def test_select_action_raises_on_empty_actions() -> None:
    policy = LinUCBPolicy()
    context = _make_context({"a": 1.0})

    with pytest.raises(ValueError, match="non-empty"):
        policy.select_action(context, [])


# --- multiple arms ---


def test_select_action_creates_one_arm_per_action() -> None:
    policy = LinUCBPolicy()
    context = _make_context({"a": 1.0, "b": 2.0})

    policy.select_action(context, _ACTIONS)

    assert set(policy.arms.keys()) == set(_ACTIONS)


def test_arms_are_independent() -> None:
    policy = LinUCBPolicy()
    context = _make_context({"a": 1.0, "b": 2.0})
    policy.select_action(context, _ACTIONS)

    policy.update(context, action="LogicCritic", reward=1.0)

    assert not np.allclose(policy.arms["LogicCritic"].b, np.zeros(2))
    for action in ("CodeCritic", "FactCritic", "MetaCritic"):
        assert np.allclose(policy.arms[action].b, np.zeros(2))


def test_arms_returns_a_copy_not_the_live_dict() -> None:
    policy = LinUCBPolicy()
    context = _make_context({"a": 1.0})
    policy.select_action(context, ["LogicCritic"])

    snapshot = policy.arms
    policy.select_action(context, ["CodeCritic"])

    assert "CodeCritic" not in snapshot
    assert "CodeCritic" in policy.arms


def test_unseen_action_gets_zero_expected_reward() -> None:
    policy = LinUCBPolicy()
    context = _make_context({"a": 1.0, "b": 1.0})

    selection = policy.select_action(context, _ACTIONS)

    assert all(p.expected_reward == 0.0 for p in selection.predictions.values())


# --- dimension consistency ---


def test_dimension_is_fixed_by_first_context() -> None:
    policy = LinUCBPolicy()
    policy.select_action(_make_context({"a": 1.0, "b": 2.0}), ["LogicCritic"])

    mismatched_context = _make_context({"a": 1.0, "b": 2.0, "c": 3.0})
    with pytest.raises(ValueError, match="dimension"):
        policy.select_action(mismatched_context, ["CodeCritic"])


def test_all_arms_created_by_the_same_policy_share_dimension() -> None:
    policy = LinUCBPolicy()
    context = _make_context({"a": 1.0, "b": 2.0, "c": 3.0})

    policy.select_action(context, _ACTIONS)

    for arm in policy.arms.values():
        assert arm.dimension == 3
        assert arm.A.shape == (3, 3)
        assert arm.A_inv.shape == (3, 3)
        assert arm.b.shape == (3,)


# --- best arm selection ---


def test_select_action_returns_lin_ucb_selection() -> None:
    policy = LinUCBPolicy()
    context = _make_context({"a": 1.0, "b": 1.0})

    selection = policy.select_action(context, _ACTIONS)

    assert isinstance(selection, LinUCBSelection)
    assert selection.selected_action in _ACTIONS
    assert set(selection.predictions.keys()) == set(_ACTIONS)


def test_selected_action_has_the_highest_upper_confidence_bound() -> None:
    policy = LinUCBPolicy()
    context = _make_context({"a": 1.0, "b": 1.0})

    selection = policy.select_action(context, _ACTIONS)

    best_ucb = max(p.upper_confidence_bound for p in selection.predictions.values())
    assert selection.predictions[selection.selected_action].upper_confidence_bound == best_ucb


def test_ties_are_broken_alphabetically() -> None:
    # All arms unseen and identical alpha/regularization => identical UCB for every action.
    policy = LinUCBPolicy()
    context = _make_context({"a": 1.0, "b": 1.0})

    selection = policy.select_action(context, _ACTIONS)

    assert selection.selected_action == sorted(_ACTIONS)[0]


def test_repeated_high_reward_makes_that_arm_preferred() -> None:
    policy = LinUCBPolicy(alpha=0.1)  # small alpha: favor exploitation over exploration
    context = _make_context({"a": 1.0, "b": 1.0})

    # Bootstrap all arms once so their confidence bounds are on comparable footing.
    for action in _ACTIONS:
        policy.update(context, action=action, reward=0.0)

    for _ in range(10):
        policy.update(context, action="CodeCritic", reward=1.0)

    selection = policy.select_action(context, _ACTIONS)

    assert selection.selected_action == "CodeCritic"
    assert selection.predictions["CodeCritic"].expected_reward > (
        selection.predictions["LogicCritic"].expected_reward
    )


def test_repeated_low_reward_makes_that_arm_unpreferred() -> None:
    policy = LinUCBPolicy(alpha=0.1)
    context = _make_context({"a": 1.0, "b": 1.0})

    for action in _ACTIONS:
        policy.update(context, action=action, reward=0.5)

    for _ in range(10):
        policy.update(context, action="MetaCritic", reward=0.0)

    selection = policy.select_action(context, _ACTIONS)

    assert selection.selected_action != "MetaCritic"


# --- update ---


def test_update_creates_arm_if_not_selected_first() -> None:
    policy = LinUCBPolicy()
    context = _make_context({"a": 1.0, "b": 1.0})

    policy.update(context, action="LogicCritic", reward=1.0)

    assert "LogicCritic" in policy.arms


def test_update_raises_on_dimension_mismatch() -> None:
    policy = LinUCBPolicy()
    policy.update(_make_context({"a": 1.0}), action="LogicCritic", reward=1.0)

    with pytest.raises(ValueError, match="dimension"):
        policy.update(_make_context({"a": 1.0, "b": 2.0}), action="CodeCritic", reward=1.0)


# --- deterministic behavior ---


def test_select_action_is_deterministic() -> None:
    context = _make_context({"a": 1.0, "b": 2.0})

    policy_1 = LinUCBPolicy()
    policy_2 = LinUCBPolicy()
    for policy in (policy_1, policy_2):
        policy.update(context, action="LogicCritic", reward=0.9)
        policy.update(context, action="CodeCritic", reward=0.1)

    selection_1 = policy_1.select_action(context, _ACTIONS)
    selection_2 = policy_2.select_action(context, _ACTIONS)

    assert selection_1.selected_action == selection_2.selected_action
    assert selection_1.predictions == selection_2.predictions


def test_select_action_does_not_mutate_arm_state() -> None:
    policy = LinUCBPolicy()
    context = _make_context({"a": 1.0, "b": 2.0})
    policy.update(context, action="LogicCritic", reward=1.0)

    a_before = policy.arms["LogicCritic"].A.copy()
    policy.select_action(context, _ACTIONS)
    a_after = policy.arms["LogicCritic"].A.copy()

    assert np.array_equal(a_before, a_after)
