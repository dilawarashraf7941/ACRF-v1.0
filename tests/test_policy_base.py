"""Unit tests for `BasePolicy` (`app/policy/base.py`).

`BasePolicy` is an abstract interface with no implementation of its own;
these tests confirm it cannot be instantiated directly and that a
subclass must implement `select_action` to be instantiable.
"""

import pytest

from app.context import ContextVector
from app.policy.base import BasePolicy
from app.policy.models import PolicyDecision


def test_cannot_instantiate_base_policy_directly() -> None:
    with pytest.raises(TypeError):
        BasePolicy()  # type: ignore[abstract]


def test_subclass_missing_select_action_cannot_be_instantiated() -> None:
    class IncompletePolicy(BasePolicy):
        pass

    with pytest.raises(TypeError):
        IncompletePolicy()  # type: ignore[abstract]


def test_subclass_implementing_select_action_can_be_instantiated() -> None:
    class MinimalPolicy(BasePolicy):
        policy_name = "MinimalPolicy"
        policy_version = "0.1.0"

        def select_action(
            self, context: ContextVector, candidate_critics: list[str]
        ) -> PolicyDecision:
            return PolicyDecision(
                selected_critics=candidate_critics[:1],
                policy_name=self.policy_name,
                policy_version=self.policy_version,
                confidence=0.0,
            )

    policy = MinimalPolicy()
    assert policy.policy_name == "MinimalPolicy"
    assert policy.policy_version == "0.1.0"


def test_default_policy_name_and_version_are_placeholders() -> None:
    assert BasePolicy.policy_name == "BasePolicy"
    assert BasePolicy.policy_version == "0.0.0"
