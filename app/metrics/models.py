"""Data models for the ACRF Metrics & Experiment Framework.

This module defines only `ExecutionMetrics` (one standardized record per
completed execution) and `ExperimentSummary` (aggregate statistics over
many `ExecutionMetrics`). It contains no reinforcement learning, no
contextual bandits, no policy optimization, and no learning of any kind.
Its only purpose is to give research experiments a stable, self-describing
way to compare runs — across the current Heuristic Policy or any future
Contextual Bandit / Offline RL / PPO / Q-learning policy — without
requiring any change to this module.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExecutionMetrics(BaseModel):
    """A standardized, immutable snapshot of evaluation metrics for one completed execution.

    Built by `MetricsCollector` (see `app/metrics/collector.py`) from an
    `AgentState`, an `ExperienceRecord`, and a `RewardSignal`. Every field
    is a plain, serializable value so this record can be consumed by
    research tooling or future learning algorithms without depending on
    any ACRF runtime type.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    execution_id: str = Field(
        ...,
        description=(
            "Identifier of the execution this metrics record describes "
            "(shared with its ExperienceRecord.experience_id)."
        ),
    )
    reward: float = Field(
        ...,
        description="The total reward computed for this execution (RewardSignal.reward).",
    )
    aggregated_quality_score: float | None = Field(
        default=None,
        description="The aggregated critic quality score for this execution, if any.",
    )
    iterations: int = Field(
        ...,
        ge=0,
        description="The number of refinement iterations completed for this execution.",
    )
    latency: float | None = Field(
        default=None,
        description="Elapsed seconds for this execution, if determinable.",
    )
    estimated_cost: float | None = Field(
        default=None,
        description="The placeholder cost proxy recorded for this execution, if any.",
    )
    selected_critics: list[str] = Field(
        default_factory=list,
        description="The critics selected for this execution.",
    )
    correction_applied: bool = Field(
        ...,
        description=(
            "Whether at least one self-correction was actually applied during this execution."
        ),
    )
    execution_status: str = Field(
        ...,
        description="The terminal execution status of this execution.",
    )
    timestamp: datetime = Field(
        ...,
        description="When this execution concluded.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Arbitrary additional data, including the policy tag used "
            "for grouping in ExperimentSummary."
        ),
    )


class ExperimentSummary(BaseModel):
    """Aggregate statistics computed over a collection of `ExecutionMetrics`.

    Built by `MetricsAggregator` (see `app/metrics/aggregator.py`).
    Averages/rates are `None` when there is no data to compute them from
    (an empty collection, or a field that was `None` on every record),
    rather than raising or reporting a misleading `0.0`.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    total_runs: int = Field(
        ...,
        ge=0,
        description="The number of ExecutionMetrics records this summary was computed from.",
    )
    average_reward: float | None = Field(
        default=None,
        description="The mean reward across all runs.",
    )
    average_quality: float | None = Field(
        default=None,
        description="The mean aggregated_quality_score across runs where it was recorded.",
    )
    average_iterations: float | None = Field(
        default=None,
        description="The mean iteration count across all runs.",
    )
    average_latency: float | None = Field(
        default=None,
        description="The mean latency across runs where it was recorded.",
    )
    average_cost: float | None = Field(
        default=None,
        description="The mean estimated_cost across runs where it was recorded.",
    )
    success_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="The fraction of runs with execution_status == 'completed'.",
    )
    correction_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="The fraction of runs where correction_applied was True.",
    )
    average_reward_per_policy: dict[str, float] = Field(
        default_factory=dict,
        description="The mean reward for each distinct policy tag observed across runs.",
    )
    critic_selection_frequency: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "How many times each critic identifier appeared across "
            "every run's selected_critics."
        ),
    )
    policy_usage: dict[str, int] = Field(
        default_factory=dict,
        description="How many runs were tagged with each distinct policy.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional diagnostic data about how this summary was computed.",
    )
