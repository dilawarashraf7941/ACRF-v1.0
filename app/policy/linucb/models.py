"""Data models for the LinUCB Contextual Bandit core.

This module defines only structured, immutable outputs
(`LinUCBPrediction`, `LinUCBSelection`) — no algorithmic logic. The
algorithm itself lives in `app/policy/linucb/arm.py` (`LinUCBArm`) and
`app/policy/linucb/policy.py` (`LinUCBPolicy`).
"""

from pydantic import BaseModel, ConfigDict, Field


class LinUCBPrediction(BaseModel):
    """One arm's LinUCB prediction for a single `ContextVector`.

    `upper_confidence_bound = expected_reward + confidence_bonus`, i.e.
    `p = θᵀx + α√(xᵀA⁻¹x)` — the value `LinUCBPolicy.select_action` ranks
    arms by.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    arm_id: str = Field(..., description="Identifier of the arm (critic) this prediction is for.")
    expected_reward: float = Field(
        ..., description="The point estimate θᵀx, with θ = A⁻¹b."
    )
    confidence_bonus: float = Field(
        ...,
        ge=0.0,
        description=(
            "The exploration term α√(xᵀA⁻¹x). Always non-negative since A⁻¹ is positive definite."
        ),
    )
    upper_confidence_bound: float = Field(
        ..., description="expected_reward + confidence_bonus — the score arms are ranked by."
    )
    context_id: str = Field(
        ..., description="The ContextVector.context_id this prediction was computed from."
    )


class LinUCBSelection(BaseModel):
    """The result of one `LinUCBPolicy.select_action` call."""

    model_config = ConfigDict(extra="allow", frozen=True)

    selected_action: str = Field(
        ...,
        description="The arm with the highest upper_confidence_bound (ties broken alphabetically).",
    )
    predictions: dict[str, LinUCBPrediction] = Field(
        ..., description="Every candidate action's LinUCBPrediction, keyed by action name."
    )
    alpha: float = Field(
        ..., ge=0.0, description="The exploration coefficient used for this selection."
    )
    context_id: str = Field(
        ..., description="The ContextVector.context_id this selection was computed from."
    )
