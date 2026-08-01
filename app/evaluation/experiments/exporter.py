"""`Exporter`: serializes `ExperimentResult`s to JSON, CSV, and Markdown.

Pure serialization only — no experiment execution and no statistics. No
reinforcement learning, no PPO, and no learning of any kind. Only the
standard library (`json`, `csv`, `io`, `pathlib`) is used; no new
dependency is introduced.
"""

import csv
import io
import json
from pathlib import Path

from app.evaluation.experiments.models import ExperimentResult

_FORMAT_BY_SUFFIX: dict[str, str] = {
    ".json": "json",
    ".csv": "csv",
    ".md": "markdown",
    ".markdown": "markdown",
}
"""Maps a file suffix to the export format `export` should use for it."""

_CSV_FIELDNAMES: tuple[str, ...] = (
    "experiment_name",
    "policy_name",
    "run_index",
    "average_reward",
    "average_quality",
    "average_latency",
    "average_iterations",
    "total_experiences",
    "match_rate",
)


class Exporter:
    """Serializes one or more `ExperimentResult`s to JSON, CSV, or Markdown.

    Stateless: every method is a pure function of the `ExperimentResult`s
    passed to it. `to_json`/`to_csv`/`to_markdown` return strings
    (testable without touching the filesystem); `export` writes one of
    those strings to a path, inferring the format from its suffix.
    """

    def to_json(self, results: list[ExperimentResult]) -> str:
        """Serialize `results` to a JSON array, full detail included (every run).

        Args:
            results: The experiment results to serialize, in order.

        Returns:
            A pretty-printed JSON string: `[{...}, {...}, ...]`.
        """
        return json.dumps([result.model_dump(mode="json") for result in results], indent=2)

    def to_csv(self, results: list[ExperimentResult]) -> str:
        """Serialize `results` to CSV, one row per (experiment, run) pair.

        This is the finest-grained export: every individual run's
        `ReplayResult` becomes its own row, so downstream analysis can
        recompute any aggregate itself rather than trusting only the
        precomputed summary (see `to_markdown` for that summary view).

        Args:
            results: The experiment results to serialize, in order.

        Returns:
            A CSV string with a header row followed by one row per run.
        """
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(_CSV_FIELDNAMES))
        writer.writeheader()
        for result in results:
            for run_index, run in enumerate(result.runs):
                writer.writerow(
                    {
                        "experiment_name": result.experiment_name,
                        "policy_name": result.policy_name,
                        "run_index": run_index,
                        "average_reward": run.average_reward,
                        "average_quality": run.average_quality,
                        "average_latency": run.average_latency,
                        "average_iterations": run.average_iterations,
                        "total_experiences": run.total_experiences,
                        "match_rate": run.metadata.get("match_rate", ""),
                    }
                )
        return buffer.getvalue()

    def to_markdown(self, results: list[ExperimentResult]) -> str:
        """Serialize `results` to a Markdown summary table, one row per experiment.

        Unlike `to_csv`, this reports each experiment's aggregate
        statistics (mean +/- standard deviation), not individual runs.

        Args:
            results: The experiment results to serialize, in order.

        Returns:
            A Markdown table string, ending in a newline.
        """
        lines = [
            "| Experiment | Policy | Runs | Avg Reward | Std Reward | "
            "Avg Quality | Avg Latency | Avg Iterations | Match Rate |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for result in results:
            lines.append(
                f"| {result.experiment_name} | {result.policy_name} | {len(result.runs)} | "
                f"{result.average_reward:.4f} | {result.std_reward:.4f} | "
                f"{result.average_quality:.4f} | {result.average_latency:.4f} | "
                f"{result.average_iterations:.4f} | {result.match_rate:.4f} |"
            )
        return "\n".join(lines) + "\n"

    def export(self, results: list[ExperimentResult], path: str | Path) -> Path:
        """Write `results` to `path`, inferring the format from its suffix.

        Args:
            results: The experiment results to export.
            path: Where to write the output. `.json` -> `to_json`,
                `.csv` -> `to_csv`, `.md`/`.markdown` -> `to_markdown`.

        Returns:
            `path`, as a `Path`.

        Raises:
            ValueError: If `path`'s suffix is not one of the supported ones.
        """
        destination = Path(path)
        format_name = _FORMAT_BY_SUFFIX.get(destination.suffix.lower())
        if format_name is None:
            raise ValueError(
                f"Unsupported export suffix {destination.suffix!r}; "
                f"use one of {sorted(_FORMAT_BY_SUFFIX)}."
            )

        if format_name == "json":
            content = self.to_json(results)
        elif format_name == "csv":
            content = self.to_csv(results)
        else:
            content = self.to_markdown(results)

        destination.write_text(content, encoding="utf-8")
        return destination
