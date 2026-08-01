"""Unit tests for `LearningReportGenerator` (`app/evaluation/learning_analysis/report.py`)."""

import csv
import io
import json

import pytest

from app.evaluation.learning_analysis.models import LearningCurve
from app.evaluation.learning_analysis.report import LearningReportGenerator, _sample_indices


def _make_curve(**overrides: object) -> LearningCurve:
    defaults: dict[str, object] = {
        "reward_per_step": [-0.325, 1.175, 1.175, 1.175],
        "cumulative_reward": [-0.325, 0.85, 2.025, 3.2],
        "instantaneous_regret": [1.5, 0.0, 0.0, 0.0],
        "cumulative_regret": [1.5, 1.5, 1.5, 1.5],
        "average_reward": 0.8,
        "moving_average_reward": [-0.325, 0.425, 0.675, 0.8],
        "metadata": {
            "num_steps": 4,
            "convergence_point": 1,
            "learning_rate_estimate": 0.45,
            "moving_average_window": 10,
            "convergence_tolerance": 0.05,
            "best_reward_observed": 1.175,
            "worst_reward_observed": -0.325,
        },
    }
    defaults.update(overrides)
    return LearningCurve(**defaults)


# --- _sample_indices ---


def test_sample_indices_of_empty_is_empty() -> None:
    assert _sample_indices(0, 10) == []


def test_sample_indices_returns_every_index_when_short() -> None:
    assert _sample_indices(5, 10) == [0, 1, 2, 3, 4]


def test_sample_indices_includes_first_and_last_when_long() -> None:
    indices = _sample_indices(1000, 10)
    assert indices[0] == 0
    assert indices[-1] == 999
    assert len(indices) <= 10


def test_sample_indices_is_sorted_and_deduplicated() -> None:
    indices = _sample_indices(37, 10)
    assert indices == sorted(set(indices))


# --- to_json ---


def test_to_json_is_valid_and_round_trips_all_fields() -> None:
    curve = _make_curve()
    parsed = json.loads(LearningReportGenerator().to_json(curve))

    assert parsed["reward_per_step"] == curve.reward_per_step
    assert parsed["metadata"]["convergence_point"] == 1


def test_to_json_of_empty_curve() -> None:
    curve = LearningCurve()
    parsed = json.loads(LearningReportGenerator().to_json(curve))
    assert parsed["reward_per_step"] == []


# --- to_csv ---


def test_to_csv_has_one_row_per_step() -> None:
    curve = _make_curve()
    rows = list(csv.DictReader(io.StringIO(LearningReportGenerator().to_csv(curve))))
    assert len(rows) == 4
    assert rows[0]["step"] == "0"
    assert rows[3]["step"] == "3"


def test_to_csv_columns_match_curve_values() -> None:
    curve = _make_curve()
    rows = list(csv.DictReader(io.StringIO(LearningReportGenerator().to_csv(curve))))

    assert float(rows[0]["reward"]) == pytest.approx(-0.325)
    assert float(rows[0]["cumulative_reward"]) == pytest.approx(-0.325)
    assert float(rows[0]["instantaneous_regret"]) == pytest.approx(1.5)
    assert float(rows[1]["reward"]) == pytest.approx(1.175)
    assert float(rows[1]["cumulative_regret"]) == pytest.approx(1.5)
    assert float(rows[3]["moving_average_reward"]) == pytest.approx(0.8)


def test_to_csv_has_expected_header() -> None:
    output = LearningReportGenerator().to_csv(_make_curve())
    header = output.splitlines()[0]
    assert header == (
        "step,reward,cumulative_reward,instantaneous_regret,cumulative_regret,"
        "moving_average_reward"
    )


def test_to_csv_of_empty_curve_has_only_header() -> None:
    output = LearningReportGenerator().to_csv(LearningCurve())
    rows = list(csv.DictReader(io.StringIO(output)))
    assert rows == []
    assert "step" in output


# --- to_markdown ---


def test_to_markdown_includes_summary_section() -> None:
    output = LearningReportGenerator().to_markdown(_make_curve())
    assert "## Summary" in output
    assert "Steps analyzed: 4" in output
    assert "Average reward: 0.800000" in output


def test_to_markdown_includes_convergence_point() -> None:
    output = LearningReportGenerator().to_markdown(_make_curve())
    assert "Convergence point: step 1" in output


def test_to_markdown_reports_na_convergence_for_empty_curve() -> None:
    output = LearningReportGenerator().to_markdown(LearningCurve())
    assert "Convergence point: n/a (no steps)" in output


def test_to_markdown_includes_learning_rate() -> None:
    output = LearningReportGenerator().to_markdown(_make_curve())
    assert "Learning rate estimate: +0.450000 reward/step" in output


def test_to_markdown_includes_sampled_steps_table() -> None:
    output = LearningReportGenerator().to_markdown(_make_curve())
    assert "## Sampled Steps" in output
    assert "| Step | Reward" in output


def test_to_markdown_of_empty_curve_has_no_data_rows() -> None:
    output = LearningReportGenerator().to_markdown(LearningCurve())
    lines = output.strip().splitlines()
    table_header_index = next(i for i, line in enumerate(lines) if line.startswith("| Step"))
    # header + separator, then nothing else
    assert lines[table_header_index + 2 :] == []


def test_to_markdown_ends_with_newline() -> None:
    output = LearningReportGenerator().to_markdown(_make_curve())
    assert output.endswith("\n")


def test_to_markdown_sampled_table_row_count_bounded_for_long_curve() -> None:
    n = 500
    curve = _make_curve(
        reward_per_step=[0.1] * n,
        cumulative_reward=[0.1 * (i + 1) for i in range(n)],
        instantaneous_regret=[0.0] * n,
        cumulative_regret=[0.0] * n,
        moving_average_reward=[0.1] * n,
        metadata={"num_steps": n, "convergence_point": 0, "learning_rate_estimate": 0.0},
    )
    output = LearningReportGenerator().to_markdown(curve)
    table_lines = [line for line in output.splitlines() if line.startswith("| ")]
    # header row + up to 10 sampled data rows
    assert len(table_lines) <= 11
