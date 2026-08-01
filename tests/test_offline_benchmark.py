"""Unit tests for `Benchmark` (`app/evaluation/offline/benchmark.py`)."""

from app.evaluation.offline.benchmark import Benchmark
from app.evaluation.offline.models import BenchmarkResult, ReplayResult


def _make_result(**overrides: object) -> ReplayResult:
    defaults: dict[str, object] = {
        "policy_name": "Policy",
        "total_experiences": 10,
        "total_reward": 5.0,
        "average_reward": 0.5,
        "average_quality": 0.6,
        "average_iterations": 1.0,
        "average_latency": 1.5,
    }
    defaults.update(overrides)
    return ReplayResult(**defaults)


def test_compare_returns_benchmark_result() -> None:
    baseline = _make_result(policy_name="HeuristicPolicy", average_reward=0.5)
    candidate = _make_result(policy_name="LinUCBPolicy", average_reward=0.6)

    result = Benchmark().compare(baseline, candidate)

    assert isinstance(result, BenchmarkResult)
    assert result.baseline_policy == "HeuristicPolicy"
    assert result.candidate_policy == "LinUCBPolicy"


def test_reward_improvement_is_candidate_minus_baseline() -> None:
    baseline = _make_result(average_reward=0.4)
    candidate = _make_result(average_reward=0.7)

    result = Benchmark().compare(baseline, candidate)

    assert result.reward_improvement == 0.7 - 0.4


def test_quality_improvement_is_candidate_minus_baseline() -> None:
    baseline = _make_result(average_quality=0.3)
    candidate = _make_result(average_quality=0.9)

    result = Benchmark().compare(baseline, candidate)

    assert result.quality_improvement == 0.9 - 0.3


def test_latency_difference_is_candidate_minus_baseline() -> None:
    baseline = _make_result(average_latency=2.0)
    candidate = _make_result(average_latency=1.0)

    result = Benchmark().compare(baseline, candidate)

    assert result.latency_difference == 1.0 - 2.0


def test_iteration_difference_is_candidate_minus_baseline() -> None:
    baseline = _make_result(average_iterations=3.0)
    candidate = _make_result(average_iterations=1.0)

    result = Benchmark().compare(baseline, candidate)

    assert result.iteration_difference == 1.0 - 3.0


def test_winner_is_candidate_when_candidate_reward_is_higher() -> None:
    baseline = _make_result(policy_name="HeuristicPolicy", average_reward=0.3)
    candidate = _make_result(policy_name="LinUCBPolicy", average_reward=0.8)

    result = Benchmark().compare(baseline, candidate)

    assert result.winner == "LinUCBPolicy"


def test_winner_is_baseline_when_baseline_reward_is_higher() -> None:
    baseline = _make_result(policy_name="HeuristicPolicy", average_reward=0.9)
    candidate = _make_result(policy_name="LinUCBPolicy", average_reward=0.2)

    result = Benchmark().compare(baseline, candidate)

    assert result.winner == "HeuristicPolicy"


def test_winner_is_tie_when_rewards_are_exactly_equal() -> None:
    baseline = _make_result(policy_name="HeuristicPolicy", average_reward=0.5)
    candidate = _make_result(policy_name="LinUCBPolicy", average_reward=0.5)

    result = Benchmark().compare(baseline, candidate)

    assert result.winner == "tie"


def test_zero_differences_when_results_are_identical() -> None:
    baseline = _make_result()
    candidate = _make_result()

    result = Benchmark().compare(baseline, candidate)

    assert result.reward_improvement == 0.0
    assert result.quality_improvement == 0.0
    assert result.latency_difference == 0.0
    assert result.iteration_difference == 0.0
    assert result.winner == "tie"


def test_metadata_records_both_results_totals() -> None:
    baseline = _make_result(policy_name="HeuristicPolicy", total_experiences=4, average_reward=0.4)
    candidate = _make_result(policy_name="LinUCBPolicy", total_experiences=6, average_reward=0.6)

    result = Benchmark().compare(baseline, candidate)

    assert result.metadata["baseline"]["total_experiences"] == 4
    assert result.metadata["candidate"]["total_experiences"] == 6


def test_compare_is_deterministic() -> None:
    baseline = _make_result(policy_name="HeuristicPolicy", average_reward=0.4)
    candidate = _make_result(policy_name="LinUCBPolicy", average_reward=0.6)

    result_a = Benchmark().compare(baseline, candidate)
    result_b = Benchmark().compare(baseline, candidate)

    assert result_a == result_b


def test_compare_does_not_mutate_inputs() -> None:
    baseline = _make_result(policy_name="HeuristicPolicy", average_reward=0.4)
    candidate = _make_result(policy_name="LinUCBPolicy", average_reward=0.6)
    baseline_dump_before = baseline.model_dump()
    candidate_dump_before = candidate.model_dump()

    Benchmark().compare(baseline, candidate)

    assert baseline.model_dump() == baseline_dump_before
    assert candidate.model_dump() == candidate_dump_before
