"""Data model for the ACRF Reward Engine.

This module defines only `RewardSignal` — a structured, immutable,
deterministic reward computed from a single `ExperienceRecord`. It
contains no reinforcement learning, no policy optimization, and no
learning of any kind. Its only purpose is to give future learning
algorithms (contextual bandits, offline RL, PPO, Q-learning, ...) a
stable, self-describing reward they can consume as-is, without requiring
any change to this module.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RewardSignal(BaseModel):
    """An immutable, deterministic reward computed from one `ExperienceRecord`.

    Built by a `BaseRewardStrategy` (see `app/reward/strategy.py`) via
    `RewardCalculator` (see `app/reward/calculator.py`). Every field is a
    plain, serializable value so this signal can be consumed by future
    learning algorithms without depending on any ACRF runtime type.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    reward: float = Field(
        ...,
        description=(
            "The total combined reward: quality_reward + completion_bonus "
            "- cost_penalty - latency_penalty - correction_penalty."
        ),
    )
    quality_reward: float = Field(
        ...,
        description="The positive component derived from aggregated_quality_score.",
    )
    efficiency_penalty: float = Field(
        ...,
        description=(
            "A rollup of cost_penalty + latency_penalty, reported for "
            "convenience. Not itself subtracted again in `reward` "
            "(its two components already are, individually)."
        ),
    )
    cost_penalty: float = Field(
        ...,
        description="The penalty magnitude derived from estimated_cost.",
    )
    latency_penalty: float = Field(
        ...,
        description="The penalty magnitude derived from latency.",
    )
    correction_penalty: float = Field(
        ...,
        description="The penalty magnitude derived from the number of correction iterations.",
    )
    completion_bonus: float = Field(
        ...,
        description=(
            "Positive for a completed execution, negative (a penalty) for "
            "a failed one, zero for any other execution_status."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "How complete the inputs to this computation were: the fraction "
            "of optional signals (aggregated_quality_score, estimated_cost, "
            "latency) that were actually present on the source ExperienceRecord."
        ),
    )
    strategy: str = Field(
        ...,
        description=(
            "Identifier of the BaseRewardStrategy implementation that produced this signal."
        ),
    )
    explanation: str = Field(
        ...,
        description="A human-readable breakdown of how `reward` was derived from its components.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Arbitrary additional diagnostic data: raw inputs, weights used, and provenance."
        ),
    )
