"""`AblationReportGenerator`: formats a list of `AblationResult`s as
Markdown, CSV, or JSON.

Pure serialization/formatting only — no ablation execution and no
statistics (those are `AblationRunner`'s job, see `runner.py`, itself
delegating to existing, unmodified components). No reinforcement
learning, no PPO, and no learning of any kind. Only the standard library
(`csv`, `io`, `json`) is used; no new dependency is introduced.
"""

import csv
import io
import json

from app.evaluation.ablation.models import AblationResult

_CSV_FIELDNAMES: tuple[str, ...] = (
    "ablation_type",
    "experiment_name",
    "baseline_policy",
    "candidate_policy",
    "baseline_reward",
    "candidate_reward",
    "reward_difference",
    "quality_difference",
    "latency_difference",
    "iteration_difference",
    "winner",
    "significant",
    "p_value",
    "effect_size",
    "conclusion",
)


def _ranked(results: list[AblationResult]) -> list[AblationResult]:
    """Return `results` sorted by `candidate_reward`, highest first."""
    return sorted(results, key=lambda result: result.candidate_reward, reverse=True)


class AblationReportGenerator:
    """Formats one or more `AblationResult`s as Markdown, CSV, or JSON.

    Stateless: every method is a pure function of the `AblationResult`s
    passed to it.
    """

    def to_json(self, results: list[AblationResult]) -> str:
        """Serialize `results` to a pretty-printed JSON array, full detail included."""
        return json.dumps([result.model_dump(mode="json") for result in results], indent=2)

    def to_csv(self, results: list[AblationResult]) -> str:
        """Serialize `results` to CSV, one row per ablation, ranked by `candidate_reward`.

        Args:
            results: The ablation results to serialize.

        Returns:
            A CSV string with a header row followed by one row per
            result, highest `candidate_reward` first.
        """
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(_CSV_FIELDNAMES))
        writer.writeheader()
        for result in _ranked(results):
            writer.writerow(
                {
                    "ablation_type": result.ablation_type,
                    "experiment_name": result.metadata.get("experiment_name", ""),
                    "baseline_policy": result.metadata.get("baseline_policy", ""),
                    "candidate_policy": result.metadata.get("candidate_policy", ""),
                    "baseline_reward": result.baseline_reward,
                    "candidate_reward": result.candidate_reward,
                    "reward_difference": result.reward_difference,
                    "quality_difference": result.quality_difference,
                    "latency_difference": result.latency_difference,
                    "iteration_difference": result.iteration_difference,
                    "winner": result.metadata.get("winner", ""),
                    "significant": result.metadata.get("significant", ""),
                    "p_value": result.metadata.get("p_value", ""),
                    "effect_size": result.metadata.get("effect_size", ""),
                    "conclusion": result.conclusion,
                }
            )
        return buffer.getvalue()

    def to_summary_table(self, results: list[AblationResult]) -> str:
        """Render a compact Markdown summary table, ranked by `candidate_reward`.

        Args:
            results: The ablation results to summarize.

        Returns:
            A Markdown table string, ending in a newline.
        """
        lines = [
            "| Ablation Type | Baseline Reward | Candidate Reward | Reward Diff | "
            "Significant | p-value |",
            "|---|---|---|---|---|---|",
        ]
        for result in _ranked(results):
            lines.append(
                f"| {result.ablation_type} | {result.baseline_reward:.4f} | "
                f"{result.candidate_reward:.4f} | {result.reward_difference:+.4f} | "
                f"{result.metadata.get('significant', 'n/a')} | "
                f"{result.metadata.get('p_value', float('nan')):.4f} |"
            )
        return "\n".join(lines) + "\n"

    def to_markdown(self, results: list[AblationResult]) -> str:
        """Render a full ablation study report: summary table, ranking, best/worst, observations.

        Args:
            results: The ablation results to report on.

        Returns:
            A Markdown-formatted string, ending in a newline.

        Raises:
            ValueError: If `results` is empty — there is nothing to rank
                or draw a "best"/"worst" configuration from.
        """
        if not results:
            raise ValueError("cannot generate a Markdown report from zero ablation results.")

        ranked = _ranked(results)
        best, worst = ranked[0], ranked[-1]

        lines = [
            "# Ablation Study Report",
            "",
            "## Summary Table",
            "",
            self.to_summary_table(results).rstrip("\n"),
            "",
            "## Ranking (by candidate reward, highest first)",
            "",
        ]
        lines.extend(
            f"{rank}. **{result.ablation_type}** "
            f"({result.metadata.get('candidate_policy', 'candidate')}): "
            f"candidate_reward = {result.candidate_reward:.4f}"
            for rank, result in enumerate(ranked, start=1)
        )
        lines.extend(
            [
                "",
                "## Best Configuration",
                "",
                f"- **{best.ablation_type}** "
                f"({best.metadata.get('candidate_policy', 'candidate')}): "
                f"candidate_reward = {best.candidate_reward:.4f}, "
                f"reward_difference = {best.reward_difference:+.4f}",
                f"- {best.conclusion}",
                "",
                "## Worst Configuration",
                "",
                f"- **{worst.ablation_type}** "
                f"({worst.metadata.get('candidate_policy', 'candidate')}): "
                f"candidate_reward = {worst.candidate_reward:.4f}, "
                f"reward_difference = {worst.reward_difference:+.4f}",
                f"- {worst.conclusion}",
                "",
                "## Key Observations",
                "",
            ]
        )
        lines.extend(self._key_observations(results))
        return "\n".join(lines) + "\n"

    @staticmethod
    def _key_observations(results: list[AblationResult]) -> list[str]:
        """Build a deterministic list of Markdown bullet points summarizing `results`."""
        significant = [r for r in results if r.metadata.get("significant")]
        improved = [r for r in results if r.reward_difference > 0]
        regressed = [r for r in results if r.reward_difference < 0]

        observations = [
            f"- {len(results)} ablation(s) evaluated; {len(significant)} showed a "
            "statistically significant reward difference.",
            f"- {len(improved)} ablation(s) improved reward over their baseline; "
            f"{len(regressed)} regressed.",
        ]

        if improved:
            best_improvement = max(improved, key=lambda r: r.reward_difference)
            observations.append(
                f"- Largest improvement: **{best_improvement.ablation_type}** "
                f"({best_improvement.reward_difference:+.4f} reward)."
            )
        if regressed:
            worst_regression = min(regressed, key=lambda r: r.reward_difference)
            observations.append(
                f"- Largest regression: **{worst_regression.ablation_type}** "
                f"({worst_regression.reward_difference:+.4f} reward)."
            )

        return observations
