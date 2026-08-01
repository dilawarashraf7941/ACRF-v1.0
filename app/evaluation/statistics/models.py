"""Data models for the ACRF Statistical Analysis Framework.

Defines only `StatisticalComparison` — a structured, immutable record of
one paired statistical comparison between two policies' experiment
results. No test computation, decision logic, or report generation
lives here; see `analyzer.py` and `report.py`. No reinforcement
learning, no PPO, and no learning of any kind.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.evaluation.experiments import ConfidenceInterval


class StatisticalComparison(BaseModel):
    """The immutable result of one paired statistical comparison.

    Built by `Analyzer.compare_samples`/`Analyzer.compare_experiments`
    (see `analyzer.py`). Every field is a plain, serializable value so
    this record can be consumed by `ReportGenerator` (see `report.py`)
    or by any future policy's evaluation, without depending on any
    ACRF runtime type beyond `ConfidenceInterval` (reused, not
    duplicated, from `app.evaluation.experiments`).
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    baseline_policy: str = Field(
        ..., description="Identifier of the reference policy being compared against."
    )
    candidate_policy: str = Field(
        ..., description="Identifier of the policy being evaluated against the baseline."
    )
    sample_size: int = Field(
        ..., ge=0, description="Number of paired observations this comparison was computed over."
    )
    mean_difference: float = Field(
        ..., description="mean(candidate values) - mean(baseline values)."
    )
    confidence_interval: ConfidenceInterval = Field(
        ...,
        description=(
            "The confidence interval for mean_difference (t-distribution based; see "
            "Analyzer.confidence_interval for why this is used regardless of test_used)."
        ),
    )
    p_value: float = Field(
        ..., ge=0.0, le=1.0, description="The p-value from whichever test test_used names."
    )
    effect_size: float = Field(
        ...,
        description=(
            "Cohen's d for paired samples (d_z = mean_difference / standard deviation of the "
            "paired differences); 0.0 when the differences have no meaningful spread to "
            "standardize against (fewer than two paired observations, or zero variance)."
        ),
    )
    test_used: str = Field(
        ...,
        description=(
            "Which test produced p_value: 'paired_t_test', 'wilcoxon_signed_rank', "
            "'degenerate_zero_variance' (every paired difference is identical), or "
            "'insufficient_data' (a single paired observation)."
        ),
    )
    significant: bool = Field(
        ..., description="Whether p_value is below the Analyzer's configured significance level."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Diagnostic data: the significance level used, the normality test's own "
            "statistic/p-value (when computed), and — for compare_experiments — the metric "
            "name and whether both experiments shared the same random_seed."
        ),
    )
