"""Unit tests for `Analyzer` (`app/evaluation/statistics/analyzer.py`).

Covers: identical samples, different samples, small sample sizes, zero
variance, confidence interval, effect size, and automatic test
selection (paired t-test vs. Wilcoxon signed-rank).
"""

import random

import pytest

from app.evaluation.experiments import ExperimentResult
from app.evaluation.offline.models import ReplayResult
from app.evaluation.statistics.analyzer import (
    DEFAULT_NORMALITY_LEVEL,
    MIN_SAMPLES_FOR_NORMALITY_TEST,
    Analyzer,
)


def _make_replay_result(average_reward: float) -> ReplayResult:
    return ReplayResult(
        policy_name="TestPolicy",
        total_experiences=5,
        total_reward=average_reward * 5,
        average_reward=average_reward,
        average_quality=0.5,
        average_iterations=1.0,
        average_latency=1.0,
    )


def _make_experiment_result(
    policy_name: str, rewards: list[float], random_seed: object = 1
) -> ExperimentResult:
    runs = [_make_replay_result(r) for r in rewards]
    return ExperimentResult(
        experiment_name=f"{policy_name}-experiment",
        policy_name=policy_name,
        runs=runs,
        average_reward=sum(rewards) / len(rewards) if rewards else 0.0,
        std_reward=0.0,
        average_quality=0.5,
        average_latency=1.0,
        average_iterations=1.0,
        match_rate=1.0,
        metadata={"random_seed": random_seed},
    )


# --- construction ---


def test_invalid_significance_level_raises() -> None:
    with pytest.raises(ValueError, match="significance_level"):
        Analyzer(significance_level=0.0)


def test_invalid_normality_level_raises() -> None:
    with pytest.raises(ValueError, match="normality_level"):
        Analyzer(normality_level=1.0)


def test_invalid_confidence_level_raises() -> None:
    with pytest.raises(ValueError, match="confidence_level"):
        Analyzer(confidence_level=-0.1)


# --- basic descriptive statistics ---


def test_mean_of_values() -> None:
    assert Analyzer.mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_mean_of_empty_is_zero() -> None:
    assert Analyzer.mean([]) == 0.0


def test_standard_deviation_of_values() -> None:
    analyzer = Analyzer()
    assert analyzer.standard_deviation([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]) == pytest.approx(
        2.13809, rel=1e-4
    )


def test_standard_deviation_of_single_value_is_zero() -> None:
    assert Analyzer().standard_deviation([5.0]) == 0.0


def test_mean_difference_is_candidate_minus_baseline() -> None:
    analyzer = Analyzer()
    assert analyzer.mean_difference([1.0, 2.0], [2.0, 4.0]) == pytest.approx(1.5)


# --- identical samples ---


def test_identical_samples_are_degenerate_zero_variance() -> None:
    analyzer = Analyzer()
    values = [0.5, 0.6, 0.7, 0.8]

    comparison = analyzer.compare_samples(values, list(values))

    assert comparison.test_used == "degenerate_zero_variance"
    assert comparison.mean_difference == 0.0
    assert comparison.p_value == 1.0
    assert comparison.significant is False
    assert comparison.effect_size == 0.0


def test_identical_samples_confidence_interval_is_a_point_at_zero() -> None:
    analyzer = Analyzer()
    values = [0.5, 0.6, 0.7, 0.8]

    comparison = analyzer.compare_samples(values, list(values))

    assert comparison.confidence_interval.lower == 0.0
    assert comparison.confidence_interval.upper == 0.0


# --- zero variance, nonzero mean (every difference identical but non-zero) ---


def test_constant_nonzero_difference_is_degenerate_and_significant() -> None:
    analyzer = Analyzer()
    baseline = [0.5, 0.5, 0.5, 0.5]
    candidate = [0.7, 0.7, 0.7, 0.7]

    comparison = analyzer.compare_samples(baseline, candidate)

    assert comparison.test_used == "degenerate_zero_variance"
    assert comparison.p_value == 0.0
    assert comparison.significant is True
    assert comparison.mean_difference == pytest.approx(0.2)


# --- different samples ---


def test_clearly_different_normal_like_samples_are_significant() -> None:
    analyzer = Analyzer()
    rng = random.Random(0)
    baseline = [rng.gauss(0.5, 0.05) for _ in range(30)]
    candidate = [v + rng.gauss(0.2, 0.05) for v in baseline]

    comparison = analyzer.compare_samples(baseline, candidate)

    assert comparison.test_used == "paired_t_test"
    assert comparison.significant is True
    assert comparison.mean_difference > 0.0
    assert comparison.p_value < 0.05


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="same length"):
        Analyzer().compare_samples([1.0, 2.0], [1.0, 2.0, 3.0])


def test_empty_samples_raise() -> None:
    with pytest.raises(ValueError, match="empty"):
        Analyzer().compare_samples([], [])


# --- small sample sizes ---


def test_single_observation_is_insufficient_data() -> None:
    comparison = Analyzer().compare_samples([0.5], [0.9])

    assert comparison.test_used == "insufficient_data"
    assert comparison.p_value == 1.0
    assert comparison.significant is False
    assert comparison.effect_size == 0.0


def test_single_observation_confidence_interval_is_a_point() -> None:
    comparison = Analyzer().compare_samples([0.5], [0.9])

    assert comparison.confidence_interval.lower == comparison.confidence_interval.upper


def test_two_observations_does_not_crash_and_skips_normality_test() -> None:
    comparison = Analyzer().compare_samples([0.5, 0.4], [0.9, 0.3])

    assert comparison.test_used == "wilcoxon_signed_rank"
    assert 0.0 <= comparison.p_value <= 1.0
    assert comparison.metadata["normality_test_skipped_reason"]


def test_below_normality_minimum_defaults_to_wilcoxon() -> None:
    analyzer = Analyzer()
    values = [1.0] * (MIN_SAMPLES_FOR_NORMALITY_TEST - 1)
    is_normal, metadata = analyzer.is_normally_distributed(values)

    assert is_normal is False
    assert "normality_test_skipped_reason" in metadata


# --- zero variance direct methods ---


def test_wilcoxon_all_zero_differences_does_not_raise() -> None:
    p_value = Analyzer.wilcoxon_signed_rank([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    assert p_value == pytest.approx(1.0)


# --- confidence interval ---


def test_confidence_interval_bounds_the_mean_difference() -> None:
    analyzer = Analyzer()
    rng = random.Random(1)
    baseline = [rng.gauss(0.5, 0.1) for _ in range(40)]
    candidate = [v + rng.gauss(0.3, 0.1) for v in baseline]

    interval = analyzer.confidence_interval(baseline, candidate)

    assert interval.lower < interval.upper
    assert interval.confidence_level == 0.95


def test_confidence_interval_respects_custom_confidence_level() -> None:
    analyzer = Analyzer()
    rng = random.Random(2)
    baseline = [rng.gauss(0.5, 0.1) for _ in range(40)]
    candidate = [v + rng.gauss(0.3, 0.1) for v in baseline]

    wide = analyzer.confidence_interval(baseline, candidate, confidence_level=0.99)
    narrow = analyzer.confidence_interval(baseline, candidate, confidence_level=0.50)

    assert (wide.upper - wide.lower) > (narrow.upper - narrow.lower)


def test_confidence_interval_of_two_values_is_computable() -> None:
    analyzer = Analyzer()
    interval = analyzer.confidence_interval([1.0, 2.0], [2.0, 5.0])
    assert interval.lower <= interval.upper


# --- effect size ---


def test_effect_size_zero_for_identical_samples() -> None:
    analyzer = Analyzer()
    assert analyzer.cohens_d([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_effect_size_is_large_for_a_strong_consistent_difference() -> None:
    analyzer = Analyzer()
    rng = random.Random(3)
    baseline = [rng.gauss(0.5, 0.02) for _ in range(20)]
    candidate = [v + 0.5 for v in baseline]  # a huge, consistent shift

    d = analyzer.cohens_d(baseline, candidate)

    assert d > 0.8  # Cohen's "large" threshold


def test_effect_size_zero_for_single_observation() -> None:
    assert Analyzer().cohens_d([0.5], [0.9]) == 0.0


def test_effect_size_sign_matches_direction_of_difference() -> None:
    analyzer = Analyzer()
    rng = random.Random(4)
    baseline = [rng.gauss(0.6, 0.05) for _ in range(20)]
    candidate = [v - 0.3 for v in baseline]  # candidate consistently worse

    assert analyzer.cohens_d(baseline, candidate) < 0.0


# --- automatic test selection ---


def test_normal_differences_select_paired_t_test() -> None:
    analyzer = Analyzer()
    rng = random.Random(10)
    baseline = [rng.gauss(0.5, 0.05) for _ in range(30)]
    candidate = [v + rng.gauss(0.1, 0.05) for v in baseline]

    comparison = analyzer.compare_samples(baseline, candidate)

    assert comparison.test_used == "paired_t_test"
    assert comparison.metadata["normality_test"] == "shapiro_wilk"
    assert comparison.metadata["normality_p_value"] > DEFAULT_NORMALITY_LEVEL


def test_skewed_differences_select_wilcoxon() -> None:
    analyzer = Analyzer()
    rng = random.Random(10)
    baseline = [rng.gauss(0.5, 0.05) for _ in range(30)]
    candidate = [b + rng.expovariate(2.0) for b in baseline]

    comparison = analyzer.compare_samples(baseline, candidate)

    assert comparison.test_used == "wilcoxon_signed_rank"
    assert comparison.metadata["normality_p_value"] < DEFAULT_NORMALITY_LEVEL


def test_is_normally_distributed_returns_metadata() -> None:
    rng = random.Random(11)
    values = [rng.gauss(0.0, 1.0) for _ in range(30)]

    is_normal, metadata = Analyzer().is_normally_distributed(values)

    assert isinstance(is_normal, bool)
    assert metadata["normality_test"] == "shapiro_wilk"
    assert "normality_p_value" in metadata


# --- compare_experiments ---


def test_compare_experiments_extracts_named_metric() -> None:
    baseline = _make_experiment_result("HeuristicPolicy", [0.5, 0.5, 0.5], random_seed=42)
    candidate = _make_experiment_result("LinUCBPolicy", [0.7, 0.7, 0.7], random_seed=42)

    comparison = Analyzer().compare_experiments(baseline, candidate, metric="average_reward")

    assert comparison.baseline_policy == "HeuristicPolicy"
    assert comparison.candidate_policy == "LinUCBPolicy"
    assert comparison.metadata["metric"] == "average_reward"
    assert comparison.sample_size == 3


def test_compare_experiments_records_same_random_seed_true() -> None:
    baseline = _make_experiment_result("HeuristicPolicy", [0.5, 0.6], random_seed=7)
    candidate = _make_experiment_result("LinUCBPolicy", [0.6, 0.7], random_seed=7)

    comparison = Analyzer().compare_experiments(baseline, candidate)

    assert comparison.metadata["same_random_seed"] is True


def test_compare_experiments_records_same_random_seed_false() -> None:
    baseline = _make_experiment_result("HeuristicPolicy", [0.5, 0.6], random_seed=7)
    candidate = _make_experiment_result("LinUCBPolicy", [0.6, 0.7], random_seed=8)

    comparison = Analyzer().compare_experiments(baseline, candidate)

    assert comparison.metadata["same_random_seed"] is False


def test_compare_experiments_mismatched_run_counts_raise() -> None:
    baseline = _make_experiment_result("HeuristicPolicy", [0.5, 0.6, 0.7])
    candidate = _make_experiment_result("LinUCBPolicy", [0.6, 0.7])

    with pytest.raises(ValueError, match="same length"):
        Analyzer().compare_experiments(baseline, candidate)


def test_compare_experiments_unknown_metric_raises() -> None:
    baseline = _make_experiment_result("HeuristicPolicy", [0.5, 0.6])
    candidate = _make_experiment_result("LinUCBPolicy", [0.6, 0.7])

    with pytest.raises(ValueError, match="no field"):
        Analyzer().compare_experiments(baseline, candidate, metric="not_a_real_field")


def test_compare_experiments_does_not_mutate_inputs() -> None:
    baseline = _make_experiment_result("HeuristicPolicy", [0.5, 0.6, 0.7])
    candidate = _make_experiment_result("LinUCBPolicy", [0.6, 0.7, 0.8])
    baseline_dump_before = baseline.model_dump()
    candidate_dump_before = candidate.model_dump()

    Analyzer().compare_experiments(baseline, candidate)

    assert baseline.model_dump() == baseline_dump_before
    assert candidate.model_dump() == candidate_dump_before


# --- determinism ---


def test_compare_samples_is_deterministic() -> None:
    analyzer = Analyzer()
    rng = random.Random(99)
    baseline = [rng.gauss(0.5, 0.1) for _ in range(20)]
    candidate = [v + rng.gauss(0.1, 0.1) for v in baseline]

    result_a = analyzer.compare_samples(list(baseline), list(candidate))
    result_b = analyzer.compare_samples(list(baseline), list(candidate))

    assert result_a == result_b
