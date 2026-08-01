"""Data model for the ACRF Learning Analysis layer.

Defines only `LearningCurve` — a structured, immutable set of derived
metrics computed from an already-completed sequence of
`app.evaluation.offline.ReplayStep`s (most meaningfully, the output of
`ReplayEngine.replay_with_learning`). No analysis logic lives here; see
`analyzer.py`. No replay, resampling, reward, or policy-update logic —
this module (and this whole package) only ever reads already-computed
data.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LearningCurve(BaseModel):
    """The immutable result of analyzing one sequential replay's step-by-step outcomes.

    Every list field is index-aligned and the same length — one entry
    per replayed step, in the order the steps occurred. Built by
    `LearningAnalyzer.analyze` (see `analyzer.py`); never constructed
    from live state, a policy, or a repository.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    reward_per_step: list[float] = Field(
        default_factory=list, description="The reward observed at each step, in order."
    )
    cumulative_reward: list[float] = Field(
        default_factory=list,
        description=(
            "Running sum of reward_per_step "
            "(cumulative_reward[i] = sum(reward_per_step[:i+1]))."
        ),
    )
    instantaneous_regret: list[float] = Field(
        default_factory=list,
        description=(
            "Per-step regret relative to the best reward observed anywhere in this "
            "run (best_reward_observed - reward_per_step[i]); always >= 0."
        ),
    )
    cumulative_regret: list[float] = Field(
        default_factory=list, description="Running sum of instantaneous_regret."
    )
    average_reward: float = Field(
        default=0.0, description="mean(reward_per_step); 0.0 for an empty run."
    )
    moving_average_reward: list[float] = Field(
        default_factory=list,
        description=(
            "A trailing moving average of reward_per_step (window size in "
            "metadata['moving_average_window']); the window shrinks near the "
            "start rather than being undefined."
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Diagnostic data: num_steps, convergence_point, learning_rate_estimate, "
            "moving_average_window, convergence_tolerance, best/worst_reward_observed."
        ),
    )

    @model_validator(mode="after")
    def _validate_series_are_aligned(self) -> "LearningCurve":
        """Ensure every per-step series has exactly the same length.

        A `LearningCurve` where these disagree could never have come
        from `LearningAnalyzer.analyze` (which always builds them from
        the same source sequence) and would silently corrupt any
        index-based consumer (e.g. `LearningReportGenerator.to_csv`).
        """
        lengths = {
            "reward_per_step": len(self.reward_per_step),
            "cumulative_reward": len(self.cumulative_reward),
            "instantaneous_regret": len(self.instantaneous_regret),
            "cumulative_regret": len(self.cumulative_regret),
            "moving_average_reward": len(self.moving_average_reward),
        }
        if len(set(lengths.values())) > 1:
            raise ValueError(f"LearningCurve series must all be the same length, got {lengths}")
        return self
