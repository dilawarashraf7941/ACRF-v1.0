"""Unit tests for `PolicyRegistry` (`app/policy/registry.py`).

Tests build their own `PolicyRegistry` instances rather than mutating the
shared `DEFAULT_POLICY_REGISTRY` singleton, to stay isolated from other
tests (see `app/policy/README.md`). `DEFAULT_POLICY_REGISTRY` itself is
covered separately, read-only, at the bottom of this file.
"""

import pytest

from app.context import ContextVector
from app.policy.base import BasePolicy
from app.policy.contextual_bandit_policy import ContextualBanditPolicy
from app.policy.heuristic_policy import HeuristicPolicy
from app.policy.models import PolicyDecision
from app.policy.registry import DEFAULT_POLICY_REGISTRY, PolicyRegistry


class _StubPolicy(BasePolicy):
    policy_name = "StubPolicy"
    policy_version = "9.9.9"

    def select_action(
        self, context: ContextVector, candidate_critics: list[str]
    ) -> PolicyDecision:
        return PolicyDecision(
            selected_critics=candidate_critics[:1],
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            confidence=0.0,
        )


class _AnotherStubPolicy(BasePolicy):
    policy_name = "AnotherStubPolicy"
    policy_version = "1.0.0"

    def select_action(
        self, context: ContextVector, candidate_critics: list[str]
    ) -> PolicyDecision:
        return PolicyDecision(
            selected_critics=[],
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            confidence=0.0,
        )


def test_register_and_get() -> None:
    registry = PolicyRegistry()
    policy = _StubPolicy()

    registry.register(policy)

    assert registry.get("StubPolicy") is policy


def test_get_unregistered_name_raises_key_error() -> None:
    registry = PolicyRegistry()

    with pytest.raises(KeyError):
        registry.get("NoSuchPolicy")


def test_list_returns_registered_names() -> None:
    registry = PolicyRegistry()
    registry.register(_StubPolicy())
    registry.register(_AnotherStubPolicy())

    assert set(registry.list()) == {"StubPolicy", "AnotherStubPolicy"}


def test_list_is_empty_for_new_registry() -> None:
    registry = PolicyRegistry()
    assert registry.list() == []


def test_first_registered_policy_becomes_default() -> None:
    registry = PolicyRegistry()
    first = _StubPolicy()
    second = _AnotherStubPolicy()

    registry.register(first)
    registry.register(second)

    assert registry.default_policy() is first


def test_explicit_default_true_overrides_first_registered() -> None:
    registry = PolicyRegistry()
    first = _StubPolicy()
    second = _AnotherStubPolicy()

    registry.register(first)
    registry.register(second, default=True)

    assert registry.default_policy() is second


def test_default_policy_raises_when_nothing_registered() -> None:
    registry = PolicyRegistry()

    with pytest.raises(ValueError):
        registry.default_policy()


def test_registering_same_name_twice_replaces_entry() -> None:
    registry = PolicyRegistry()
    first = _StubPolicy()
    replacement = _StubPolicy()

    registry.register(first)
    registry.register(replacement)

    assert registry.get("StubPolicy") is replacement
    assert registry.list() == ["StubPolicy"]


def test_default_policy_registry_default_is_heuristic_policy() -> None:
    assert isinstance(DEFAULT_POLICY_REGISTRY.default_policy(), HeuristicPolicy)


def test_default_policy_registry_lists_both_shipped_policies() -> None:
    assert set(DEFAULT_POLICY_REGISTRY.list()) == {"HeuristicPolicy", "ContextualBanditPolicy"}


def test_default_policy_registry_contextual_bandit_is_registered_but_not_default() -> None:
    policy = DEFAULT_POLICY_REGISTRY.get("ContextualBanditPolicy")
    assert isinstance(policy, ContextualBanditPolicy)
    assert not isinstance(DEFAULT_POLICY_REGISTRY.default_policy(), ContextualBanditPolicy)
