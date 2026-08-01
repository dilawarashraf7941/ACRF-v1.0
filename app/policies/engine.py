"""Policy engine infrastructure for ACRF: state, actions, scores, and the
abstract scoring interface a future adaptive policy engine will implement.

This module defines only infrastructure — data models and an abstract
interface. It contains no adaptive decision-making and no learning. The
one concrete class provided, `PlaceholderPolicyEngine`, implements the
`PolicyEngine` interface with a fixed, neutral, non-adaptive scoring
function so the interface is usable and testable before a real scoring
algorithm exists.

Deliberately decoupled from `AgentState` (see `app/state/state.py`) and
from `AdaptivePolicy` (see `app/policies/models.py`), so this
infrastructure remains independently reusable; translating those into a
`PolicyState` is left to a future integration layer.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PolicyState(BaseModel):
    """A minimal, self-contained snapshot of the signals a policy engine
    needs to score and select among candidate critic actions.
    """

    model_config = ConfigDict(extra="allow")

    session_id: str = Field(
        ...,
        description="Identifier of the session this policy state belongs to.",
    )
    task_id: str = Field(
        ...,
        description="Identifier of the task this policy state belongs to.",
    )
    task_type: str | None = Field(
        default=None,
        description="Open-ended classification of the task, if known.",
    )
    iteration_count: int = Field(
        default=0,
        ge=0,
        description="Number of refinement iterations completed so far.",
    )
    max_iterations: int = Field(
        default=10,
        ge=0,
        description="Upper bound on the number of iterations permitted.",
    )
    critic_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Scores already produced by critics, keyed by critic identifier.",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional context the policy engine may consider (e.g. error features, risk signals).",
    )


class CriticActionType(str, Enum):
    """A small, closed taxonomy of the kinds of actions a policy engine can choose among."""

    INVOKE_CRITIC = "invoke_critic"
    SKIP_CRITIC = "skip_critic"
    INVOKE_META_CRITIC = "invoke_meta_critic"
    REQUEST_SELF_CORRECTION = "request_self_correction"
    FINALIZE = "finalize"


class CriticAction(BaseModel):
    """A single candidate action available to the policy engine, concerning critic usage."""

    model_config = ConfigDict(extra="allow")

    action_type: CriticActionType = Field(
        ...,
        description="The kind of action this represents.",
    )
    critic_id: str | None = Field(
        default=None,
        description="Identifier of the critic this action pertains to, if applicable.",
    )
    rationale: str | None = Field(
        default=None,
        description="Optional human-readable explanation for why this action is being considered.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional data describing this action.",
    )


class PolicyScore(BaseModel):
    """A score assigned to a single candidate `CriticAction` by a `PolicyEngine`."""

    model_config = ConfigDict(extra="allow")

    action: CriticAction = Field(
        ...,
        description="The candidate action this score applies to.",
    )
    score: float = Field(
        ...,
        description="The numeric score assigned to the action. The scale is defined by the scoring implementation, not this schema.",
    )
    rationale: str | None = Field(
        default=None,
        description="Optional human-readable explanation for the score.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional data supporting the score.",
    )


class PolicyEngine(ABC):
    """Abstract interface for a policy engine that scores candidate critic actions.

    This defines only the contract; no scoring algorithm, adaptive
    decision-making, or learning is implemented here. See
    `PlaceholderPolicyEngine` for a deterministic, non-adaptive
    placeholder implementation of this interface.
    """

    @abstractmethod
    def score(self, state: PolicyState, action: CriticAction) -> PolicyScore:
        """Score a single candidate action given the current policy state.

        This is the engine's abstract scoring API. Concrete subclasses
        must implement the actual scoring algorithm; no scoring
        intelligence, adaptation, or learning is implemented here.

        Args:
            state: The current policy state to score the action against.
            action: The candidate action to score.

        Returns:
            A `PolicyScore` describing how the engine scores this action.
        """
        raise NotImplementedError


class PlaceholderPolicyEngine(PolicyEngine):
    """A deterministic, non-adaptive placeholder implementation of `PolicyEngine`.

    Always returns a fixed, neutral score of `0.0` for every action,
    regardless of `state` or `action`. This performs no adaptive
    decision-making and no learning; it exists solely to provide a
    working, testable implementation of the `PolicyEngine` interface
    until a real scoring algorithm is implemented.
    """

    def score(self, state: PolicyState, action: CriticAction) -> PolicyScore:
        """Return a fixed, neutral `PolicyScore` of `0.0` for `action`.

        Args:
            state: Accepted to satisfy the `PolicyEngine` interface; not
                consulted by this placeholder.
            action: The candidate action being scored.

        Returns:
            A `PolicyScore` with `score=0.0` and a rationale noting that
            no scoring logic is implemented.
        """
        return PolicyScore(
            action=action,
            score=0.0,
            rationale="Placeholder scoring function: no scoring logic implemented.",
            metadata={"engine": "PlaceholderPolicyEngine"},
        )
