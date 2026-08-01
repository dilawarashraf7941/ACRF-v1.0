"""Data models for the ACRF Experiment Framework.

Defines only structured, immutable outputs and configuration — no
experiment execution, statistics, or export logic. `ExperimentRunner`
(see `runner.py`) consumes `ExperimentConfig` and produces
`ExperimentResult`; `Analyzer` (see `analyzer.py`) produces
`StatisticalSummary`/`ConfidenceInterval` along the way. No
reinforcement learning, no PPO, no online/live learning, and no learning
of any kind is implemented anywhere in this module.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.evaluation.offline import ReplayResult


class ExperimentConfig(BaseModel):
    """A single, reproducible experiment specification.

    Fully determines an `ExperimentRunner.run` call: which policy, with
    which hyperparameters, replayed how many times, against which
    candidate action set, and — for `num_runs > 1` — seeded so the exact
    same bootstrap resamples are drawn every time this config is run
    against the same source data (see `runner.py`'s module docstring for
    why resampling is how "N independent runs" produce genuinely
    different results without mutating or duplicating any replay logic).
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    experiment_name: str = Field(
        ..., description="A human-readable identifier for this experiment."
    )
    policy_name: str = Field(
        ...,
        description=(
            "Which policy to run. The built-in policy factory supports "
            "'HeuristicPolicy' and 'LinUCBPolicy'; a custom policy_factory "
            "injected into ExperimentRunner can support any other name."
        ),
    )
    alpha: float | None = Field(
        default=None,
        description=(
            "LinUCB exploration coefficient, used only when policy_name is "
            "'LinUCBPolicy'. Ignored for 'HeuristicPolicy'. Defaults to "
            "LinUCBPolicy's own default (1.0) when unset."
        ),
    )
    random_seed: int = Field(
        ...,
        description=(
            "Seeds every bootstrap resample this experiment draws (only "
            "relevant when num_runs > 1). An identical config replayed "
            "against identical source data always produces an identical "
            "ExperimentResult."
        ),
    )
    num_runs: int = Field(
        ...,
        ge=1,
        description=(
            "Number of independent runs. 1 replays the source data "
            "directly (no resampling). >1 draws that many independent "
            "bootstrap resamples (with replacement, seeded by "
            "random_seed) of the source data, one per run."
        ),
    )
    candidate_actions: list[str] | None = Field(
        default=None,
        description=(
            "The fixed candidate action set every replayed experience is "
            "scored against. Defaults to ReplayEngine's own "
            "DEFAULT_CANDIDATE_CRITICS when unset."
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary additional configuration/provenance data."
    )


class ConfidenceInterval(BaseModel):
    """A `[lower, upper]` interval at a given confidence level."""

    model_config = ConfigDict(extra="allow", frozen=True)

    lower: float = Field(..., description="The interval's lower bound.")
    upper: float = Field(..., description="The interval's upper bound.")
    confidence_level: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="The confidence level this interval was computed at.",
    )


class StatisticalSummary(BaseModel):
    """Mean, standard deviation, min, max, and a confidence interval over a set of values."""

    model_config = ConfigDict(extra="allow", frozen=True)

    mean: float = Field(..., description="The arithmetic mean.")
    std_dev: float = Field(
        ..., description="The sample standard deviation (ddof=1); 0.0 for fewer than 2 values."
    )
    minimum: float = Field(..., description="The minimum value.")
    maximum: float = Field(..., description="The maximum value.")
    confidence_interval: ConfidenceInterval = Field(
        ...,
        description=(
            "The empirical percentile confidence interval (see Analyzer.confidence_interval)."
        ),
    )
    sample_size: int = Field(
        ..., ge=0, description="Number of values this summary was computed over."
    )


class ExperimentResult(BaseModel):
    """Aggregate statistics from running one `ExperimentConfig` end to end."""

    model_config = ConfigDict(extra="allow", frozen=True)

    experiment_name: str = Field(..., description="Echoes ExperimentConfig.experiment_name.")
    policy_name: str = Field(..., description="Echoes ExperimentConfig.policy_name.")
    runs: list[ReplayResult] = Field(
        default_factory=list,
        description="One ReplayResult per independent run, in run order, for full traceability.",
    )
    average_reward: float = Field(
        ..., description="Mean of each run's ReplayResult.average_reward."
    )
    std_reward: float = Field(
        ...,
        description="Sample standard deviation of each run's average_reward; 0.0 for a single run.",
    )
    average_quality: float = Field(
        ..., description="Mean of each run's ReplayResult.average_quality."
    )
    average_latency: float = Field(
        ..., description="Mean of each run's ReplayResult.average_latency."
    )
    average_iterations: float = Field(
        ..., description="Mean of each run's ReplayResult.average_iterations."
    )
    match_rate: float = Field(
        ..., description="Mean of each run's ReplayResult.metadata['match_rate']."
    )
    critic_selection_frequency: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Mean, across runs, of each critic's ReplayResult.critic_selection_frequency "
            "(a critic absent from a given run's frequency contributes 0.0 for that run)."
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Diagnostic data: random_seed, num_runs, and the reward "
            "distribution's confidence interval/min/max."
        ),
    )
