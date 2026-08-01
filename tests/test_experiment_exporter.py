"""Unit tests for `Exporter` (`app/evaluation/experiments/exporter.py`)."""

import csv
import io
import json

import pytest

from app.evaluation.experiments.exporter import Exporter
from app.evaluation.experiments.models import ExperimentResult
from app.evaluation.offline.models import ReplayResult


def _make_replay_result(**overrides: object) -> ReplayResult:
    defaults: dict[str, object] = {
        "policy_name": "HeuristicPolicy",
        "total_experiences": 3,
        "total_reward": 1.5,
        "average_reward": 0.5,
        "average_quality": 0.6,
        "average_iterations": 1.0,
        "average_latency": 1.5,
        "metadata": {"match_rate": 0.75},
    }
    defaults.update(overrides)
    return ReplayResult(**defaults)


def _make_experiment_result(**overrides: object) -> ExperimentResult:
    defaults: dict[str, object] = {
        "experiment_name": "baseline",
        "policy_name": "HeuristicPolicy",
        "runs": [
            _make_replay_result(average_reward=0.4),
            _make_replay_result(average_reward=0.6),
        ],
        "average_reward": 0.5,
        "std_reward": 0.1,
        "average_quality": 0.6,
        "average_latency": 1.5,
        "average_iterations": 1.0,
        "match_rate": 0.75,
        "critic_selection_frequency": {"CodeCritic": 1.0},
    }
    defaults.update(overrides)
    return ExperimentResult(**defaults)


# --- to_json ---


def test_to_json_produces_parseable_array() -> None:
    result = _make_experiment_result()

    output = Exporter().to_json([result])
    parsed = json.loads(output)

    assert isinstance(parsed, list)
    assert len(parsed) == 1
    assert parsed[0]["experiment_name"] == "baseline"


def test_to_json_includes_full_run_detail() -> None:
    result = _make_experiment_result()

    parsed = json.loads(Exporter().to_json([result]))

    assert len(parsed[0]["runs"]) == 2
    assert parsed[0]["runs"][0]["average_reward"] == 0.4


def test_to_json_of_multiple_results() -> None:
    results = [
        _make_experiment_result(experiment_name="baseline"),
        _make_experiment_result(experiment_name="candidate"),
    ]

    parsed = json.loads(Exporter().to_json(results))

    assert [entry["experiment_name"] for entry in parsed] == ["baseline", "candidate"]


def test_to_json_of_empty_list() -> None:
    assert json.loads(Exporter().to_json([])) == []


# --- to_csv ---


def test_to_csv_has_one_row_per_run() -> None:
    result = _make_experiment_result()

    output = Exporter().to_csv([result])
    rows = list(csv.DictReader(io.StringIO(output)))

    assert len(rows) == 2
    assert rows[0]["experiment_name"] == "baseline"
    assert rows[0]["run_index"] == "0"
    assert rows[1]["run_index"] == "1"


def test_to_csv_row_values_match_run_fields() -> None:
    result = _make_experiment_result()

    rows = list(csv.DictReader(io.StringIO(Exporter().to_csv([result]))))

    assert float(rows[0]["average_reward"]) == pytest.approx(0.4)
    assert float(rows[1]["average_reward"]) == pytest.approx(0.6)


def test_to_csv_of_multiple_results_concatenates_rows() -> None:
    results = [
        _make_experiment_result(experiment_name="baseline"),
        _make_experiment_result(experiment_name="candidate"),
    ]

    rows = list(csv.DictReader(io.StringIO(Exporter().to_csv(results))))

    assert len(rows) == 4
    assert {row["experiment_name"] for row in rows} == {"baseline", "candidate"}


def test_to_csv_of_empty_list_has_only_header() -> None:
    output = Exporter().to_csv([])
    rows = list(csv.DictReader(io.StringIO(output)))
    assert rows == []
    assert "experiment_name" in output


def test_to_csv_of_result_with_no_runs_produces_no_rows() -> None:
    result = _make_experiment_result(runs=[])
    rows = list(csv.DictReader(io.StringIO(Exporter().to_csv([result]))))
    assert rows == []


# --- to_markdown ---


def test_to_markdown_has_a_header_and_one_row_per_experiment() -> None:
    results = [
        _make_experiment_result(experiment_name="baseline"),
        _make_experiment_result(experiment_name="candidate"),
    ]

    output = Exporter().to_markdown(results)
    lines = output.strip().splitlines()

    assert lines[0].startswith("| Experiment")
    assert lines[1].startswith("|---")
    assert len(lines) == 4  # header + separator + 2 experiments


def test_to_markdown_reports_aggregate_statistics() -> None:
    result = _make_experiment_result(average_reward=0.5, std_reward=0.1)

    output = Exporter().to_markdown([result])

    assert "0.5000" in output
    assert "0.1000" in output


def test_to_markdown_of_empty_list_has_only_header() -> None:
    output = Exporter().to_markdown([])
    lines = output.strip().splitlines()
    assert len(lines) == 2


# --- export (file writing) ---


def test_export_writes_json_file(tmp_path) -> None:
    result = _make_experiment_result()
    destination = tmp_path / "results.json"

    written_path = Exporter().export([result], destination)

    assert written_path == destination
    assert json.loads(destination.read_text(encoding="utf-8"))[0]["experiment_name"] == "baseline"


def test_export_writes_csv_file(tmp_path) -> None:
    result = _make_experiment_result()
    destination = tmp_path / "results.csv"

    Exporter().export([result], destination)

    rows = list(csv.DictReader(io.StringIO(destination.read_text(encoding="utf-8"))))
    assert len(rows) == 2


def test_export_writes_markdown_file(tmp_path) -> None:
    result = _make_experiment_result()
    destination = tmp_path / "results.md"

    Exporter().export([result], destination)

    content = destination.read_text(encoding="utf-8")
    assert content.startswith("| Experiment")


def test_export_accepts_markdown_alias_suffix(tmp_path) -> None:
    result = _make_experiment_result()
    destination = tmp_path / "results.markdown"

    Exporter().export([result], destination)

    assert destination.read_text(encoding="utf-8").startswith("| Experiment")


def test_export_rejects_unsupported_suffix(tmp_path) -> None:
    result = _make_experiment_result()
    destination = tmp_path / "results.txt"

    with pytest.raises(ValueError, match="Unsupported export suffix"):
        Exporter().export([result], destination)


def test_export_accepts_string_path(tmp_path) -> None:
    result = _make_experiment_result()
    destination = str(tmp_path / "results.json")

    written_path = Exporter().export([result], destination)

    assert written_path.exists()
