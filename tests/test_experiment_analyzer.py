"""Unit tests for `Analyzer` (`app/evaluation/experiments/analyzer.py`)."""

import pytest

from app.evaluation.experiments.analyzer import Analyzer
from app.evaluation.offline.models import ReplayResult


def _make_replay_result(**overrides: object) -> ReplayResult:
    defaults: dict[str, object] = {
        "policy_name": "HeuristicPolicy",
        "total_experiences": 2,
        "total_reward": 1.0,
        "average_reward": 0.5,
        "average_quality": 0.6,
        "average_iterations": 1.0,
        "average_latency": 1.5,
    }
    defaults.update(overrides)
    return ReplayResult(**defaults)


# --- mean ---


def test_mean_of_values() -> None:
    assert Analyzer().mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_mean_of_empty_sequence_is_zero() -> None:
    assert Analyzer().mean([]) == 0.0


# --- std_dev ---


def test_std_dev_of_values() -> None:
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    assert Analyzer().std_dev(values) == pytest.approx(2.13809, rel=1e-4)


def test_std_dev_of_single_value_is_zero() -> None:
    assert Analyzer().std_dev([5.0]) == 0.0


def test_std_dev_of_empty_sequence_is_zero() -> None:
    assert Analyzer().std_dev([]) == 0.0


def test_std_dev_of_identical_values_is_zero() -> None:
    assert Analyzer().std_dev([3.0, 3.0, 3.0]) == pytest.approx(0.0)


# --- minimum / maximum ---


def test_minimum_of_values() -> None:
    assert Analyzer().minimum([3.0, 1.0, 2.0]) == 1.0


def test_maximum_of_values() -> None:
    assert Analyzer().maximum([3.0, 1.0, 2.0]) == 3.0


def test_minimum_of_empty_sequence_is_zero() -> None:
    assert Analyzer().minimum([]) == 0.0


def test_maximum_of_empty_sequence_is_zero() -> None:
    assert Analyzer().maximum([]) == 0.0


# --- confidence_interval ---


def test_confidence_interval_of_empty_sequence_is_zero_width() -> None:
    interval = Analyzer().confidence_interval([])
    assert interval.lower == 0.0
    assert interval.upper == 0.0


def test_confidence_interval_of_single_value_degenerates_to_that_value() -> None:
    interval = Analyzer().confidence_interval([4.2])
    assert interval.lower == 4.2
    assert interval.upper == 4.2


def test_confidence_interval_bounds_the_values() -> None:
    values = [float(v) for v in range(1, 101)]  # 1..100

    interval = Analyzer().confidence_interval(values)

    assert interval.lower >= min(values)
    assert interval.upper <= max(values)
    assert interval.lower < interval.upper


def test_confidence_interval_default_confidence_level_is_95_percent() -> None:
    interval = Analyzer().confidence_interval([1.0, 2.0, 3.0, 4.0, 5.0])
    assert interval.confidence_level == 0.95


def test_confidence_interval_respects_custom_confidence_level() -> None:
    values = [float(v) for v in range(1, 101)]

    wide_interval = Analyzer().confidence_interval(values, confidence_level=0.99)
    narrow_interval = Analyzer().confidence_interval(values, confidence_level=0.50)

    wide_width = wide_interval.upper - wide_interval.lower
    narrow_width = narrow_interval.upper - narrow_interval.lower
    assert wide_width > narrow_width


# --- summarize ---


def test_summarize_combines_all_statistics() -> None:
    summary = Analyzer().summarize([1.0, 2.0, 3.0, 4.0, 5.0])

    assert summary.mean == pytest.approx(3.0)
    assert summary.minimum == 1.0
    assert summary.maximum == 5.0
    assert summary.sample_size == 5
    assert summary.std_dev > 0.0
    assert summary.confidence_interval.confidence_level == 0.95


def test_summarize_of_empty_sequence_is_all_zero() -> None:
    summary = Analyzer().summarize([])

    assert summary.mean == 0.0
    assert summary.std_dev == 0.0
    assert summary.minimum == 0.0
    assert summary.maximum == 0.0
    assert summary.sample_size == 0


# --- trends ---


def test_reward_trend_returns_average_reward_in_order() -> None:
    runs = [_make_replay_result(average_reward=0.1), _make_replay_result(average_reward=0.9)]
    assert Analyzer().reward_trend(runs) == [0.1, 0.9]


def test_quality_trend_returns_average_quality_in_order() -> None:
    runs = [_make_replay_result(average_quality=0.2), _make_replay_result(average_quality=0.8)]
    assert Analyzer().quality_trend(runs) == [0.2, 0.8]


def test_latency_trend_returns_average_latency_in_order() -> None:
    runs = [_make_replay_result(average_latency=1.0), _make_replay_result(average_latency=2.0)]
    assert Analyzer().latency_trend(runs) == [1.0, 2.0]


def test_trends_of_empty_runs_are_empty_lists() -> None:
    analyzer = Analyzer()
    assert analyzer.reward_trend([]) == []
    assert analyzer.quality_trend([]) == []
    assert analyzer.latency_trend([]) == []
