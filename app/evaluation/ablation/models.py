"""Data models for the ACRF Ablation Study Framework.

Defines only `AblationConfig` and `AblationResult` — no ablation
execution, statistics, or report generation lives here; see `runner.py`
and `report.py`. No reinforcement learning, no PPO, and no learning of
any kind.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AblationConfig(BaseModel):
    """Fully specifies one ablation comparison: a baseline arm vs. a candidate arm.

    `ablation_type` selects which of `AblationRunner`'s supported
    recipes builds the two arms (see `runner.py` and `README.md` for the
    exact meaning of each type and what `metadata` key each one reads).
    `baseline_policy`/`candidate_policy` are the policy identifiers used
    both to build the arms (for most ablation types) and to label the
    resulting `AblationResult`.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    experiment_name: str = Field(..., description="A human-readable identifier for this ablation.")
    baseline_policy: str = Field(
        ..., description="The reference arm's policy identifier, e.g. 'HeuristicPolicy'."
    )
    candidate_policy: str = Field(
        ..., description="The ablated arm's policy identifier, e.g. 'LinUCBPolicy'."
    )
    ablation_type: str = Field(
        ...,
        description=(
            "Which ablation recipe to run: 'no_exploration', 'alpha_sweep', "
            "'random_critic_selection', 'heuristic_only', 'linucb_only', "
            "'reduced_context_features', or 'alternative_reward_definitions'."
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Ablation-type-specific parameters (e.g. 'alpha', "
            "'keep_feature_fraction') and arbitrary provenance data."
        ),
    )


class AblationResult(BaseModel):
    """The immutable outcome of running one `AblationConfig`."""

    model_config = ConfigDict(extra="allow", frozen=True)

    ablation_type: str = Field(..., description="Echoes AblationConfig.ablation_type.")
    baseline_reward: float = Field(
        ..., description="The baseline arm's ExperimentResult.average_reward."
    )
    candidate_reward: float = Field(
        ..., description="The candidate arm's ExperimentResult.average_reward."
    )
    reward_difference: float = Field(
        ..., description="candidate_reward - baseline_reward (from Benchmark.reward_improvement)."
    )
    quality_difference: float = Field(
        ..., description="candidate average_quality - baseline average_quality (from Benchmark)."
    )
    latency_difference: float = Field(
        ..., description="candidate average_latency - baseline average_latency (from Benchmark)."
    )
    iteration_difference: float = Field(
        ...,
        description="candidate average_iterations - baseline average_iterations (from Benchmark).",
    )
    conclusion: str = Field(
        ...,
        description=(
            "A plain-language sentence combining the Benchmark winner and the "
            "Statistics Analyzer's significance/effect-size verdict."
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Diagnostic data: baseline_policy, candidate_policy, winner, "
            "p_value, effect_size, test_used, significant, sample_size, "
            "num_runs, random_seed, and every AblationConfig.metadata key."
        ),
    )
