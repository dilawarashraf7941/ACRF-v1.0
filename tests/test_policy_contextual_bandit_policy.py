"""Unit tests for the `ContextualBanditPolicy` stub (`app/policy/contextual_bandit_policy.py`).

No contextual bandit algorithm is implemented; these tests confirm the
stub is correctly registered as a `BasePolicy` and always raises
`NotImplementedError` when actually invoked.
"""

import pytest

from app.context import ContextVector
from app.policy.base import BasePolicy
from app.policy.contextual_bandit_policy import ContextualBanditPolicy


def _make_context() -> ContextVector:
    return ContextVector(context_id="ctx-1", timestamp="2024-01-01T00:00:00Z")


def test_is_a_base_policy() -> None:
    assert isinstance(ContextualBanditPolicy(), BasePolicy)


def test_has_expected_name_and_version() -> None:
    policy = ContextualBanditPolicy()
    assert policy.policy_name == "ContextualBanditPolicy"
    assert policy.policy_version == "0.0.0"


def test_select_action_raises_not_implemented_error() -> None:
    policy = ContextualBanditPolicy()

    with pytest.raises(NotImplementedError):
        policy.select_action(_make_context(), ["LogicCritic", "CodeCritic"])


def test_select_action_raises_even_with_empty_candidates() -> None:
    policy = ContextualBanditPolicy()

    with pytest.raises(NotImplementedError):
        policy.select_action(_make_context(), [])


def test_not_implemented_error_message_mentions_no_learning() -> None:
    policy = ContextualBanditPolicy()

    with pytest.raises(NotImplementedError, match="not implemented"):
        policy.select_action(_make_context(), ["LogicCritic"])
