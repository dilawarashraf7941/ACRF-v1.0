"""Data models for the ACRF critic infrastructure.

This module defines the structured result a single critic produces
(`CriticResult`) and the structured result of combining several critics'
results into one (`AggregatedCriticResult`). It contains no evaluation
logic, no aggregation algorithms, no LLM calls, no routing, and no
adaptive policy behavior.

Deliberately decoupled from `AgentState` (see `app/state/state.py`) and
from the lighter-weight `CriticFeedback` model declared there, so this
infrastructure remains independently reusable; translating a
`CriticResult`/`AggregatedCriticResult` into
`AgentState.critic_feedback`/`critic_scores`/`aggregated_quality_score` is
left to a future integration layer.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CriticType(str, Enum):
    """A small, closed taxonomy of the built-in critic categories."""

    LOGIC = "logic"
    CODE = "code"
    FACT = "fact"
    META = "meta"
    CUSTOM = "custom"


class CriticResult(BaseModel):
    """The structured result produced by a single critic evaluation."""

    model_config = ConfigDict(extra="allow")

    critic_name: str = Field(
        ...,
        description="Identifier of the critic that produced this result (e.g. 'LogicCritic').",
    )
    critic_type: CriticType = Field(
        default=CriticType.CUSTOM,
        description="The category of critic that produced this result.",
    )
    score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Normalized quality score, from 0.0 to 1.0, assigned by the critic to the evaluated content.",
    )
    passed: bool | None = Field(
        default=None,
        description="Whether the critic considers the evaluated content acceptable. `None` means not determined.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="The critic's confidence, from 0.0 to 1.0, in its own result.",
    )
    feedback: str | None = Field(
        default=None,
        description="Optional human-readable explanation of the result.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional data produced by the critic.",
    )


class AggregatedCriticResult(BaseModel):
    """The structured result of combining multiple `CriticResult`s into one."""

    model_config = ConfigDict(extra="allow")

    strategy_name: str = Field(
        ...,
        description="Identifier of the aggregation strategy that produced this result (e.g. 'MajorityVoteStrategy').",
    )
    aggregated_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Normalized overall quality score, from 0.0 to 1.0, produced by the aggregation strategy.",
    )
    aggregated_passed: bool | None = Field(
        default=None,
        description="Whether the aggregation strategy considers the evaluated content acceptable overall. `None` means not determined.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="The aggregation strategy's confidence, from 0.0 to 1.0, in its own result.",
    )
    contributing_critics: list[str] = Field(
        default_factory=list,
        description="Identifiers of the critics whose results were considered by this aggregation.",
    )
    individual_results: list[CriticResult] = Field(
        default_factory=list,
        description="The original, per-critic `CriticResult`s that were aggregated, preserved for traceability.",
    )
    rationale: str | None = Field(
        default=None,
        description="Optional human-readable explanation of the aggregated result.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional data produced by the aggregation strategy.",
    )
