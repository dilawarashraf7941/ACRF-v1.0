"""Data model for the ACRF Context Encoding Layer.

This module defines only `ContextVector` — a structured, immutable,
deterministic numeric representation of an `AgentState` (and, optionally,
a matching `ExperienceRecord`). It contains no reinforcement learning, no
contextual bandits, no policy optimization, and no learning of any kind.
Its only purpose is to give a future Contextual Bandit / Offline RL / PPO
/ Q-learning policy a stable, ready-made input representation to consume
as-is, without requiring any change to this module.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ContextVector(BaseModel):
    """An immutable, deterministic numeric encoding of one `AgentState`.

    Built by `ContextEncoder` (see `app/context/encoder.py`) and,
    optionally, rescaled by `ContextNormalizer` (see
    `app/context/normalizer.py`), which returns a *new* `ContextVector`
    with `normalized=True` rather than mutating this one in place.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    context_id: str = Field(
        ...,
        description=(
            "Deterministic identifier for this context, derived from "
            "(session_id, task_id, iterations). Identical inputs always "
            "produce the identical id; distinct inputs produce distinct ids."
        ),
    )
    source_execution_id: str | None = Field(
        default=None,
        description=(
            "The ExperienceRecord.experience_id this context was "
            "additionally derived from, if any."
        ),
    )
    features: dict[str, float] = Field(
        default_factory=dict,
        description="The named numeric features that make up this context, in `feature_order`.",
    )
    feature_order: list[str] = Field(
        default_factory=list,
        description=(
            "The canonical, stable ordering of `features`' keys, so a "
            "consumer can build a fixed-width array without depending on dict iteration order."
        ),
    )
    normalized: bool = Field(
        default=False,
        description="Whether `features` have been rescaled by a ContextNormalizer.",
    )
    normalization_strategy: str | None = Field(
        default=None,
        description="Identifier of the normalization strategy applied, if `normalized` is True.",
    )
    timestamp: datetime = Field(
        ...,
        description=(
            "When the source execution's data was captured (read from "
            "AgentState/ExperienceRecord, never independently generated)."
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Arbitrary additional data, including any outcome-derived "
            "features (see app/context/encoder.py) kept separate from "
            "`features` so they are never mistaken for pre-decision context."
        ),
    )
