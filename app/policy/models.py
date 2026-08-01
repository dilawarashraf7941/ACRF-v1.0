"""Data model for the ACRF Policy Abstraction Layer.

This module defines only `PolicyDecision` — a structured, immutable
output produced by any `BasePolicy` implementation (see
`app/policy/base.py`). It contains no contextual bandit algorithms, no
reinforcement learning, no policy learning, and no reward/experience
updates. Its only purpose is to give a uniform, self-describing decision
shape that every current and future policy (`HeuristicPolicy`,
`ContextualBanditPolicy`, an eventual `OfflineRLPolicy`/`OnlineRLPolicy`)
produces identically, so callers never need to know which concrete
policy produced a given decision.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PolicyDecision(BaseModel):
    """The immutable output of a single `BasePolicy.select_action` call."""

    model_config = ConfigDict(extra="allow", frozen=True)

    selected_critics: list[str] = Field(
        ...,
        description="The critic identifiers this policy selected from the candidate set.",
    )
    scores: dict[str, float] = Field(
        default_factory=dict,
        description="The score this policy assigned to each candidate critic.",
    )
    ranking: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "The candidates ordered highest score first, as plain dicts "
            "(critic_name/score/rank), matching CriticRanking.as_list_of_dicts()."
        ),
    )
    policy_name: str = Field(
        ...,
        description="Identifier of the BasePolicy implementation that produced this decision.",
    )
    policy_version: str = Field(
        ...,
        description="Version identifier of the policy implementation that produced this decision.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How confident this policy is in its decision. Meaning is policy-defined.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional diagnostic data specific to the producing policy.",
    )
