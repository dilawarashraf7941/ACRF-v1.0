"""Data models for the ACRF Offline Replay & Benchmark Framework.

Defines only structured, immutable outputs — no replay, evaluation, or
benchmarking logic. `ReplayEngine` (see `replay.py`) produces `ReplayStep`
entries; `OfflineEvaluator` (see `evaluator.py`) aggregates them into a
`ReplayResult`; `Benchmark` (see `benchmark.py`) compares two
`ReplayResult`s into a `BenchmarkResult`. No reinforcement learning, no
PPO, no online/live learning, and no learning of any kind is implemented
anywhere in this module.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ReplayStep(BaseModel):
    """One stored `ExperienceRecord` a policy's replayed decision matched.

    Built by `ReplayEngine.replay` (see `replay.py`) for every stored
    experience where the policy being replayed selected the same
    critic(s) actually recorded for that experience — the standard
    "replay method" for offline (off-policy) evaluation of logged bandit
    feedback (Li et al., 2011, "Unbiased Offline Evaluation of
    Contextual-bandit-based News Article Recommendation Algorithms"; see
    `replay.py`'s module docstring for exactly which of that paper's
    guarantees do, and do not, apply to ACRF's non-randomized logging
    policy). Experiences the policy would *not* have selected are never
    turned into a `ReplayStep`: fabricating a reward for an action that
    was never actually taken would bias the evaluation.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    experience_id: str = Field(
        ..., description="The source ExperienceRecord.experience_id this step replays."
    )
    context_id: str = Field(
        ...,
        description=(
            "The offline-replay ContextVector.context_id built from the "
            "source experience (see build_offline_context_vector)."
        ),
    )
    selected_critics: list[str] = Field(
        ..., description="The critic(s) the replayed policy selected for this experience."
    )
    reward: float = Field(
        ...,
        description=(
            "RewardCalculator.calculate(experience).reward — the recorded "
            "experience's reward, replayed, never recomputed from a live run."
        ),
    )
    quality: float | None = Field(
        default=None, description="experience.aggregated_quality_score, if recorded."
    )
    iterations: int = Field(..., ge=0, description="experience.iterations.")
    latency: float | None = Field(default=None, description="experience.latency, if recorded.")


class ReplayResult(BaseModel):
    """Aggregate statistics from replaying every matched experience for one policy."""

    model_config = ConfigDict(extra="allow", frozen=True)

    policy_name: str = Field(..., description="Identifier of the policy that was replayed.")
    total_experiences: int = Field(
        ...,
        ge=0,
        description=(
            "Number of stored experiences this policy's replayed decisions "
            "matched (see ReplayStep). The denominator for every average below."
        ),
    )
    total_reward: float = Field(..., description="Sum of reward across every matched experience.")
    average_reward: float = Field(
        ..., description="total_reward / total_experiences, or 0.0 if none matched."
    )
    average_quality: float = Field(
        ...,
        description=(
            "Mean aggregated_quality_score across matched experiences "
            "(a missing score contributes 0.0, matching app/reward's convention)."
        ),
    )
    average_iterations: float = Field(
        ..., description="Mean iterations across matched experiences."
    )
    average_latency: float = Field(
        ...,
        description=(
            "Mean latency across matched experiences (a missing latency "
            "contributes 0.0, matching app/reward's convention)."
        ),
    )
    critic_selection_frequency: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Fraction of matched experiences (0.0-1.0) in which each "
            "critic was selected by this policy."
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Diagnostic data: total stored experiences, match rate, etc.",
    )


class BenchmarkResult(BaseModel):
    """The outcome of comparing two `ReplayResult`s: a baseline and a candidate policy."""

    model_config = ConfigDict(extra="allow", frozen=True)

    baseline_policy: str = Field(..., description="The baseline ReplayResult.policy_name.")
    candidate_policy: str = Field(..., description="The candidate ReplayResult.policy_name.")
    reward_improvement: float = Field(
        ..., description="candidate.average_reward - baseline.average_reward."
    )
    quality_improvement: float = Field(
        ..., description="candidate.average_quality - baseline.average_quality."
    )
    latency_difference: float = Field(
        ...,
        description=(
            "candidate.average_latency - baseline.average_latency "
            "(negative means the candidate was faster)."
        ),
    )
    iteration_difference: float = Field(
        ...,
        description=(
            "candidate.average_iterations - baseline.average_iterations "
            "(negative means the candidate needed fewer iterations)."
        ),
    )
    winner: str = Field(
        ...,
        description=(
            "Whichever policy_name achieved the higher average_reward, or "
            "'tie' if the two are exactly equal."
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Diagnostic data: both ReplayResults' raw totals, for traceability.",
    )
