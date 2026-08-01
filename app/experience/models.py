"""Data model for the ACRF Experience Memory Layer.

This module defines only `ExperienceRecord` — a structured, immutable
snapshot of a single completed execution. It contains no learning, no
scoring, no replay-buffer logic, and no training code. Its only purpose
is to give future adaptive-learning algorithms a stable, self-describing
record they can consume without requiring any change to this module.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExperienceRecord(BaseModel):
    """An immutable, structured snapshot of a single completed ACRF execution.

    Built by `ExperienceRecorder` (see `app/experience/recorder.py`) from
    an `AgentState`, and stored by `evaluation_node` into both
    `state.memory_context["experience"]` and an `ExperienceRepository`.
    Every field here is a plain, serializable value so this record can be
    consumed by future learning algorithms, or persisted by a future
    repository backend, without depending on any ACRF runtime type.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    experience_id: str = Field(
        ...,
        description=(
            "Deterministic identifier for this experience, derived from "
            "(session_id, task_id, iterations). Identical inputs always "
            "produce the identical id; distinct inputs produce distinct ids."
        ),
    )
    session_id: str = Field(
        ...,
        description="Identifier of the session this experience belongs to.",
    )
    task_id: str = Field(
        ...,
        description="Identifier of the task this experience belongs to.",
    )
    timestamp: datetime = Field(
        ...,
        description=(
            "When this execution concluded, read from "
            "AgentState.execution_metadata.updated_at."
        ),
    )
    state_features: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "A read-only snapshot of relevant AgentState signals "
            "(task type, error features, planner output, etc.)."
        ),
    )
    selected_critics: list[str] = Field(
        default_factory=list,
        description="The critics selected for this execution (AgentState.selected_critics).",
    )
    critic_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Per-critic scores recorded during this execution (AgentState.critic_scores).",
    )
    aggregated_quality_score: float | None = Field(
        default=None,
        description="The aggregated critic quality score for this execution, if any.",
    )
    correction_decision: dict[str, Any] | None = Field(
        default=None,
        description=(
            "The correction policy decision recorded during this execution, "
            "if any (see app/correction_policy)."
        ),
    )
    iterations: int = Field(
        ...,
        ge=0,
        description="The number of refinement iterations completed for this execution.",
    )
    final_response: str | None = Field(
        default=None,
        description="The final response produced by this execution, if any.",
    )
    execution_status: str = Field(
        ...,
        description=(
            "The terminal execution status of this execution "
            "(AgentState.execution_status.value)."
        ),
    )
    latency: float | None = Field(
        default=None,
        description="Elapsed seconds between execution start and completion, if determinable.",
    )
    estimated_cost: float | None = Field(
        default=None,
        description=(
            "A placeholder cost proxy (currently the sum of worker token "
            "usage). No pricing model exists yet in this framework; "
            "consumers should not treat this as a real monetary cost."
        ),
    )
    memory_usage: dict[str, Any] = Field(
        default_factory=dict,
        description="A read-only snapshot of memory-subsystem signals present on AgentState.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional provenance data about how this experience was recorded.",
    )
