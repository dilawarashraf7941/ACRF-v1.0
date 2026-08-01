"""`ReportGenerator`: turns `StatisticalComparison`s into human- or machine-readable reports.

Pure serialization/formatting only — no statistical computation (that is
`Analyzer`'s job, see `analyzer.py`). No reinforcement learning, no PPO,
and no learning of any kind. Only the standard library (`json`) is used;
no new dependency is introduced.
"""

import json

from app.evaluation.statistics.models import StatisticalComparison

_EFFECT_SIZE_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (0.2, "negligible"),
    (0.5, "small"),
    (0.8, "medium"),
)
"""Cohen's conventional |d| thresholds, in ascending order."""

_LARGE_EFFECT_LABEL = "large"
"""The label used when |d| meets or exceeds the largest threshold in `_EFFECT_SIZE_THRESHOLDS`."""


def _interpret_effect_size(effect_size: float) -> str:
    """Map an effect size to Cohen's conventional negligible/small/medium/large label."""
    magnitude = abs(effect_size)
    for threshold, label in _EFFECT_SIZE_THRESHOLDS:
        if magnitude < threshold:
            return label
    return _LARGE_EFFECT_LABEL


def _direction(mean_difference: float, candidate_policy: str, baseline_policy: str) -> str:
    """Describe, in words, which policy the mean difference favors."""
    if mean_difference > 0:
        return f"{candidate_policy} scored higher than {baseline_policy} on average"
    if mean_difference < 0:
        return f"{candidate_policy} scored lower than {baseline_policy} on average"
    return f"{candidate_policy} and {baseline_policy} scored identically on average"


class ReportGenerator:
    """Formats a `StatisticalComparison` as Markdown, JSON, or a compact summary table.

    Stateless: every method is a pure function of the
    `StatisticalComparison`(s) passed to it.
    """

    def to_markdown(self, comparison: StatisticalComparison) -> str:
        """Render a full narrative Markdown report for one comparison.

        Includes a hypothesis statement, which test was used, the
        p-value, the effect size (with its Cohen's-d interpretation),
        the confidence interval, a plain-language interpretation, and a
        conclusion.

        Args:
            comparison: The comparison to report on.

        Returns:
            A Markdown-formatted string, ending in a newline.
        """
        metric = comparison.metadata.get("metric", "the measured outcome")
        significance_level = comparison.metadata.get("significance_level", 0.05)
        effect_label = _interpret_effect_size(comparison.effect_size)
        direction = _direction(
            comparison.mean_difference, comparison.candidate_policy, comparison.baseline_policy
        )
        confidence_pct = int(round(comparison.confidence_interval.confidence_level * 100))

        if comparison.significant:
            conclusion = (
                f"The difference IS statistically significant "
                f"(p = {comparison.p_value:.4f} < alpha = {significance_level})."
            )
        else:
            conclusion = (
                f"The difference is NOT statistically significant "
                f"(p = {comparison.p_value:.4f} >= alpha = {significance_level})."
            )

        title = f"{comparison.candidate_policy} vs {comparison.baseline_policy}"
        lines = [
            f"# Statistical Comparison: {title}",
            "",
            "## Hypothesis",
            (
                f"- H0 (null): There is no difference in {metric} between "
                f"{comparison.baseline_policy} and {comparison.candidate_policy}."
            ),
            (
                f"- H1 (alternative): There is a difference in {metric} between "
                f"{comparison.baseline_policy} and {comparison.candidate_policy}."
            ),
            "",
            "## Test",
            f"- Test used: **{comparison.test_used}**",
            f"- Sample size: {comparison.sample_size}",
            f"- p-value: {comparison.p_value:.6f}",
            f"- Significant at alpha = {significance_level}: **{comparison.significant}**",
            "",
            "## Effect",
            f"- Mean difference: {comparison.mean_difference:.6f} ({direction})",
            f"- Effect size (Cohen's d, paired): {comparison.effect_size:.6f} ({effect_label})",
            (
                f"- {confidence_pct}% confidence interval: "
                f"[{comparison.confidence_interval.lower:.6f}, "
                f"{comparison.confidence_interval.upper:.6f}]"
            ),
            "",
            "## Interpretation",
            f"- {direction}, a {effect_label} effect size.",
            "",
            "## Conclusion",
            f"- {conclusion}",
        ]
        return "\n".join(lines) + "\n"

    def to_json(self, comparison: StatisticalComparison) -> str:
        """Serialize `comparison` to a pretty-printed JSON string."""
        return json.dumps(comparison.model_dump(mode="json"), indent=2)

    def to_summary_table(self, comparisons: list[StatisticalComparison]) -> str:
        """Render a compact Markdown summary table, one row per comparison.

        Args:
            comparisons: The comparisons to summarize, in order.

        Returns:
            A Markdown table string, ending in a newline.
        """
        lines = [
            "| Baseline | Candidate | n | Test | p-value | Effect Size | Significant | Mean Diff |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for comparison in comparisons:
            lines.append(
                f"| {comparison.baseline_policy} | {comparison.candidate_policy} | "
                f"{comparison.sample_size} | {comparison.test_used} | "
                f"{comparison.p_value:.4f} | {comparison.effect_size:.4f} | "
                f"{comparison.significant} | {comparison.mean_difference:.4f} |"
            )
        return "\n".join(lines) + "\n"
