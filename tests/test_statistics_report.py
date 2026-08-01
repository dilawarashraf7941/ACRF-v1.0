"""Unit tests for `ReportGenerator` (`app/evaluation/statistics/report.py`)."""

import json

import pytest

from app.evaluation.experiments import ConfidenceInterval
from app.evaluation.statistics.models import StatisticalComparison
from app.evaluation.statistics.report import ReportGenerator


def _make_comparison(**overrides: object) -> StatisticalComparison:
    defaults: dict[str, object] = {
        "baseline_policy": "HeuristicPolicy",
        "candidate_policy": "LinUCBPolicy",
        "sample_size": 20,
        "mean_difference": 0.15,
        "confidence_interval": ConfidenceInterval(lower=0.05, upper=0.25, confidence_level=0.95),
        "p_value": 0.01,
        "effect_size": 0.9,
        "test_used": "paired_t_test",
        "significant": True,
        "metadata": {"metric": "average_reward", "significance_level": 0.05},
    }
    defaults.update(overrides)
    return StatisticalComparison(**defaults)


# --- to_markdown ---


def test_to_markdown_includes_hypothesis() -> None:
    output = ReportGenerator().to_markdown(_make_comparison())
    assert "## Hypothesis" in output
    assert "H0" in output
    assert "H1" in output


def test_to_markdown_includes_test_used() -> None:
    output = ReportGenerator().to_markdown(_make_comparison(test_used="wilcoxon_signed_rank"))
    assert "wilcoxon_signed_rank" in output


def test_to_markdown_includes_p_value() -> None:
    output = ReportGenerator().to_markdown(_make_comparison(p_value=0.0123))
    assert "0.012300" in output


def test_to_markdown_includes_effect_size_and_interpretation() -> None:
    output = ReportGenerator().to_markdown(_make_comparison(effect_size=0.9))
    assert "0.900000" in output
    assert "large" in output


def test_to_markdown_effect_size_interpretation_thresholds() -> None:
    generator = ReportGenerator()
    assert "negligible" in generator.to_markdown(_make_comparison(effect_size=0.05))
    assert "small" in generator.to_markdown(_make_comparison(effect_size=0.3))
    assert "medium" in generator.to_markdown(_make_comparison(effect_size=0.6))
    assert "large" in generator.to_markdown(_make_comparison(effect_size=1.2))


def test_to_markdown_includes_conclusion_when_significant() -> None:
    output = ReportGenerator().to_markdown(_make_comparison(significant=True, p_value=0.01))
    assert "## Conclusion" in output
    assert "IS statistically significant" in output


def test_to_markdown_includes_conclusion_when_not_significant() -> None:
    output = ReportGenerator().to_markdown(
        _make_comparison(significant=False, p_value=0.8, test_used="degenerate_zero_variance")
    )
    assert "is NOT statistically significant" in output


def test_to_markdown_reports_direction_of_difference() -> None:
    higher = ReportGenerator().to_markdown(_make_comparison(mean_difference=0.5))
    lower = ReportGenerator().to_markdown(_make_comparison(mean_difference=-0.5))

    assert "scored higher than" in higher
    assert "scored lower than" in lower


def test_to_markdown_falls_back_to_generic_metric_label() -> None:
    output = ReportGenerator().to_markdown(_make_comparison(metadata={}))
    assert "the measured outcome" in output


def test_to_markdown_ends_with_newline() -> None:
    output = ReportGenerator().to_markdown(_make_comparison())
    assert output.endswith("\n")


# --- to_json ---


def test_to_json_round_trips_all_fields() -> None:
    comparison = _make_comparison()
    parsed = json.loads(ReportGenerator().to_json(comparison))

    assert parsed["baseline_policy"] == "HeuristicPolicy"
    assert parsed["candidate_policy"] == "LinUCBPolicy"
    assert parsed["test_used"] == "paired_t_test"
    assert parsed["confidence_interval"]["lower"] == pytest.approx(0.05)


def test_to_json_is_valid_json() -> None:
    output = ReportGenerator().to_json(_make_comparison())
    json.loads(output)  # raises if invalid


# --- to_summary_table ---


def test_to_summary_table_has_header_and_one_row_per_comparison() -> None:
    comparisons = [
        _make_comparison(candidate_policy="LinUCBPolicy-a"),
        _make_comparison(candidate_policy="LinUCBPolicy-b"),
    ]

    output = ReportGenerator().to_summary_table(comparisons)
    lines = output.strip().splitlines()

    assert lines[0].startswith("| Baseline")
    assert lines[1].startswith("|---")
    assert len(lines) == 4


def test_to_summary_table_of_empty_list_has_only_header() -> None:
    output = ReportGenerator().to_summary_table([])
    lines = output.strip().splitlines()
    assert len(lines) == 2


def test_to_summary_table_reports_key_statistics() -> None:
    comparison = _make_comparison(p_value=0.0321, effect_size=0.77, mean_difference=0.123)
    output = ReportGenerator().to_summary_table([comparison])

    assert "0.0321" in output
    assert "0.7700" in output
    assert "0.1230" in output
