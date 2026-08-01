"""Unit tests for `AblationReportGenerator` (`app/evaluation/ablation/report.py`)."""

import csv
import io
import json

import pytest

from app.evaluation.ablation.models import AblationResult
from app.evaluation.ablation.report import AblationReportGenerator


def _make_result(**overrides: object) -> AblationResult:
    defaults: dict[str, object] = {
        "ablation_type": "linucb_only",
        "baseline_reward": 0.5,
        "candidate_reward": 0.6,
        "reward_difference": 0.1,
        "quality_difference": 0.05,
        "latency_difference": -0.2,
        "iteration_difference": -1.0,
        "conclusion": "LinUCBPolicy performed significantly better (p=0.0100).",
        "metadata": {
            "experiment_name": "study-linucb-only",
            "baseline_policy": "HeuristicPolicy",
            "candidate_policy": "LinUCBPolicy",
            "winner": "LinUCBPolicy",
            "significant": True,
            "p_value": 0.01,
            "effect_size": 0.9,
        },
    }
    defaults.update(overrides)
    return AblationResult(**defaults)


# --- to_json ---


def test_to_json_produces_parseable_array() -> None:
    results = [_make_result()]
    parsed = json.loads(AblationReportGenerator().to_json(results))
    assert isinstance(parsed, list)
    assert parsed[0]["ablation_type"] == "linucb_only"


def test_to_json_of_empty_list() -> None:
    assert json.loads(AblationReportGenerator().to_json([])) == []


# --- to_csv ---


def test_to_csv_has_one_row_per_result() -> None:
    results = [
        _make_result(ablation_type="linucb_only", candidate_reward=0.6),
        _make_result(ablation_type="heuristic_only", candidate_reward=0.4),
    ]

    rows = list(csv.DictReader(io.StringIO(AblationReportGenerator().to_csv(results))))

    assert len(rows) == 2
    assert {row["ablation_type"] for row in rows} == {"linucb_only", "heuristic_only"}


def test_to_csv_is_ranked_by_candidate_reward_descending() -> None:
    results = [
        _make_result(ablation_type="low", candidate_reward=0.2),
        _make_result(ablation_type="high", candidate_reward=0.9),
        _make_result(ablation_type="mid", candidate_reward=0.5),
    ]

    rows = list(csv.DictReader(io.StringIO(AblationReportGenerator().to_csv(results))))

    assert [row["ablation_type"] for row in rows] == ["high", "mid", "low"]


def test_to_csv_of_empty_list_has_only_header() -> None:
    output = AblationReportGenerator().to_csv([])
    rows = list(csv.DictReader(io.StringIO(output)))
    assert rows == []
    assert "ablation_type" in output


# --- to_summary_table ---


def test_to_summary_table_has_header_and_one_row_per_result() -> None:
    results = [_make_result(ablation_type="a"), _make_result(ablation_type="b")]
    lines = AblationReportGenerator().to_summary_table(results).strip().splitlines()
    assert lines[0].startswith("| Ablation Type")
    assert lines[1].startswith("|---")
    assert len(lines) == 4


# --- to_markdown ---


def test_to_markdown_includes_all_required_sections() -> None:
    results = [
        _make_result(ablation_type="a"),
        _make_result(ablation_type="b", candidate_reward=0.9),
    ]
    output = AblationReportGenerator().to_markdown(results)

    assert "## Summary Table" in output
    assert "## Ranking" in output
    assert "## Best Configuration" in output
    assert "## Worst Configuration" in output
    assert "## Key Observations" in output


def test_to_markdown_best_configuration_has_highest_candidate_reward() -> None:
    results = [
        _make_result(ablation_type="low", candidate_reward=0.2),
        _make_result(ablation_type="high", candidate_reward=0.9),
    ]

    output = AblationReportGenerator().to_markdown(results)
    best_section = output.split("## Best Configuration")[1].split("## Worst Configuration")[0]

    assert "high" in best_section
    assert "low" not in best_section


def test_to_markdown_worst_configuration_has_lowest_candidate_reward() -> None:
    results = [
        _make_result(ablation_type="low", candidate_reward=0.2),
        _make_result(ablation_type="high", candidate_reward=0.9),
    ]

    output = AblationReportGenerator().to_markdown(results)
    worst_section = output.split("## Worst Configuration")[1].split("## Key Observations")[0]

    assert "low" in worst_section


def test_to_markdown_key_observations_report_counts() -> None:
    results = [
        _make_result(
            ablation_type="improved", reward_difference=0.2, metadata={"significant": True}
        ),
        _make_result(
            ablation_type="regressed", reward_difference=-0.1, metadata={"significant": False}
        ),
    ]

    output = AblationReportGenerator().to_markdown(results)
    observations = output.split("## Key Observations")[1]

    assert "2 ablation(s) evaluated" in observations
    assert "1 showed a statistically significant" in observations
    assert "1 ablation(s) improved reward" in observations
    assert "1 regressed" in observations
    assert "Largest improvement" in observations
    assert "Largest regression" in observations


def test_to_markdown_raises_on_empty_results() -> None:
    with pytest.raises(ValueError, match="zero ablation results"):
        AblationReportGenerator().to_markdown([])


def test_to_markdown_ranking_is_ordered_by_candidate_reward() -> None:
    results = [
        _make_result(ablation_type="low", candidate_reward=0.1),
        _make_result(ablation_type="high", candidate_reward=0.9),
        _make_result(ablation_type="mid", candidate_reward=0.5),
    ]

    output = AblationReportGenerator().to_markdown(results)
    ranking_section = output.split("## Ranking")[1].split("## Best Configuration")[0]

    high_index = ranking_section.index("high")
    mid_index = ranking_section.index("mid")
    low_index = ranking_section.index("low")
    assert high_index < mid_index < low_index


def test_to_markdown_ends_with_newline() -> None:
    output = AblationReportGenerator().to_markdown([_make_result()])
    assert output.endswith("\n")
