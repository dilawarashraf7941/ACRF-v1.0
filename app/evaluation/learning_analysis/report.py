"""`LearningReportGenerator`: formats a `LearningCurve` as Markdown, CSV, or JSON.

Pure serialization/formatting only — no analysis (that is
`LearningAnalyzer`'s job, see `analyzer.py`). No replay, reward, or
policy logic. Only the standard library (`json`, `csv`, `io`) is used;
no new dependency is introduced.
"""

import csv
import io
import json

from app.evaluation.learning_analysis.models import LearningCurve

_CSV_FIELDNAMES: tuple[str, ...] = (
    "step",
    "reward",
    "cumulative_reward",
    "instantaneous_regret",
    "cumulative_regret",
    "moving_average_reward",
)

_MAX_MARKDOWN_SAMPLE_POINTS = 10
"""How many rows `to_markdown`'s sampled-steps table shows, regardless of run length."""


def _sample_indices(length: int, max_points: int) -> list[int]:
    """Pick up to `max_points` indices spanning `range(length)`, including the first and last.

    Deterministic and duplicate-free: for `length <= max_points`, returns
    every index; otherwise returns an evenly-spaced subset.
    """
    if length == 0:
        return []
    if length <= max_points:
        return list(range(length))
    if max_points <= 1:
        return [length - 1]
    step = (length - 1) / (max_points - 1)
    indices = {round(i * step) for i in range(max_points)}
    return sorted(indices)


class LearningReportGenerator:
    """Formats a `LearningCurve` as Markdown, CSV, or JSON.

    Stateless: every method is a pure function of the `LearningCurve`
    passed to it. `to_csv`/`to_json` export full per-step detail — the
    data `Figures: ... Reward Curve, Cumulative Reward, Instantaneous
    Regret, Cumulative Regret, Moving Average Reward` would be plotted
    from; `to_markdown` is a human-readable summary with a compact
    sampled preview, not a full per-step dump.
    """

    def to_json(self, curve: LearningCurve) -> str:
        """Serialize `curve` to a pretty-printed JSON object, full detail included."""
        return json.dumps(curve.model_dump(mode="json"), indent=2)

    def to_csv(self, curve: LearningCurve) -> str:
        """Serialize `curve` to CSV, one row per step.

        Columns: `step, reward, cumulative_reward, instantaneous_regret,
        cumulative_regret, moving_average_reward` — exactly the data
        needed to plot each of the five figures this module supports
        exporting data for.

        Args:
            curve: The learning curve to serialize.

        Returns:
            A CSV string with a header row followed by one row per step.
        """
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(_CSV_FIELDNAMES))
        writer.writeheader()
        for i in range(len(curve.reward_per_step)):
            writer.writerow(
                {
                    "step": i,
                    "reward": curve.reward_per_step[i],
                    "cumulative_reward": curve.cumulative_reward[i],
                    "instantaneous_regret": curve.instantaneous_regret[i],
                    "cumulative_regret": curve.cumulative_regret[i],
                    "moving_average_reward": curve.moving_average_reward[i],
                }
            )
        return buffer.getvalue()

    def to_markdown(self, curve: LearningCurve) -> str:
        """Render a Markdown summary: key metrics plus a compact sampled-steps table.

        Args:
            curve: The learning curve to report on.

        Returns:
            A Markdown-formatted string, ending in a newline.
        """
        n = len(curve.reward_per_step)
        final_cumulative_reward = curve.cumulative_reward[-1] if n else 0.0
        final_cumulative_regret = curve.cumulative_regret[-1] if n else 0.0
        convergence_point = curve.metadata.get("convergence_point")
        convergence_label = (
            f"step {convergence_point}" if convergence_point is not None else "n/a (no steps)"
        )
        learning_rate = curve.metadata.get("learning_rate_estimate", 0.0)
        best_reward = curve.metadata.get("best_reward_observed", 0.0)
        worst_reward = curve.metadata.get("worst_reward_observed", 0.0)

        lines = [
            "# Learning Curve Report",
            "",
            "## Summary",
            f"- Steps analyzed: {n}",
            f"- Average reward: {curve.average_reward:.6f}",
            f"- Best / worst reward observed: {best_reward:.6f} / {worst_reward:.6f}",
            f"- Final cumulative reward: {final_cumulative_reward:.6f}",
            f"- Final cumulative regret: {final_cumulative_regret:.6f}",
            f"- Convergence point: {convergence_label}",
            f"- Learning rate estimate: {learning_rate:+.6f} reward/step",
            "",
            "## Sampled Steps",
            "",
            "| Step | Reward | Cumulative Reward | Instantaneous Regret | "
            "Cumulative Regret | Moving Avg Reward |",
            "|---|---|---|---|---|---|",
        ]
        for i in _sample_indices(n, _MAX_MARKDOWN_SAMPLE_POINTS):
            lines.append(
                f"| {i} | {curve.reward_per_step[i]:.4f} | {curve.cumulative_reward[i]:.4f} | "
                f"{curve.instantaneous_regret[i]:.4f} | {curve.cumulative_regret[i]:.4f} | "
                f"{curve.moving_average_reward[i]:.4f} |"
            )
        return "\n".join(lines) + "\n"
