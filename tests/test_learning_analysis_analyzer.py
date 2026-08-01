"""Unit tests for `LearningAnalyzer` (`app/evaluation/learning_analysis/analyzer.py`).

Covers: reward accumulation, regret calculation, convergence detection,
moving average, and determinism.
"""

import pytest

from app.evaluation.learning_analysis.analyzer import LearningAnalyzer
from app.evaluation.offline.models import ReplayStep


def _make_step(experience_id: str, reward: float) -> ReplayStep:
    return ReplayStep(
        experience_id=experience_id,
        context_id=f"ctx-{experience_id}",
        selected_critics=["CodeCritic"],
        reward=reward,
        iterations=0,
    )


def _make_steps(rewards: list[float]) -> list[ReplayStep]:
    return [_make_step(f"exp-{i}", reward) for i, reward in enumerate(rewards)]


# --- construction ---


def test_invalid_moving_average_window_raises() -> None:
    with pytest.raises(ValueError, match="moving_average_window"):
        LearningAnalyzer(moving_average_window=0)


@pytest.mark.parametrize("tolerance", [0.0, 1.0, -0.1, 1.5])
def test_invalid_convergence_tolerance_raises(tolerance: float) -> None:
    with pytest.raises(ValueError, match="convergence_tolerance"):
        LearningAnalyzer(convergence_tolerance=tolerance)


# --- reward accumulation ---


def test_reward_per_step_extracts_rewards_in_order() -> None:
    steps = _make_steps([0.1, 0.5, 0.3])
    assert LearningAnalyzer.reward_per_step(steps) == [0.1, 0.5, 0.3]


def test_reward_per_step_of_empty_is_empty() -> None:
    assert LearningAnalyzer.reward_per_step([]) == []


def test_cumulative_reward_is_running_sum() -> None:
    assert LearningAnalyzer.cumulative_reward([1.0, 2.0, 3.0]) == [1.0, 3.0, 6.0]


def test_cumulative_reward_of_empty_is_empty() -> None:
    assert LearningAnalyzer.cumulative_reward([]) == []


def test_cumulative_reward_handles_negative_values() -> None:
    assert LearningAnalyzer.cumulative_reward([-1.0, 2.0, -0.5]) == [-1.0, 1.0, 0.5]


def test_analyze_end_to_end_reward_fields_from_real_steps() -> None:
    steps = _make_steps([0.2, 0.4, 0.6])
    curve = LearningAnalyzer().analyze(steps)

    assert curve.reward_per_step == [0.2, 0.4, 0.6]
    assert curve.cumulative_reward == pytest.approx([0.2, 0.6, 1.2])
    assert curve.average_reward == pytest.approx(0.4)
    assert curve.metadata["num_steps"] == 3


def test_analyze_empty_steps_degrades_gracefully() -> None:
    curve = LearningAnalyzer().analyze([])

    assert curve.reward_per_step == []
    assert curve.cumulative_reward == []
    assert curve.instantaneous_regret == []
    assert curve.cumulative_regret == []
    assert curve.moving_average_reward == []
    assert curve.average_reward == 0.0
    assert curve.metadata["convergence_point"] is None
    assert curve.metadata["learning_rate_estimate"] == 0.0


# --- regret calculation ---


def test_instantaneous_regret_relative_to_best_observed() -> None:
    rewards = [0.5, 1.0, 0.2]
    regret = LearningAnalyzer.instantaneous_regret(rewards)
    assert regret == pytest.approx([0.5, 0.0, 0.8])


def test_instantaneous_regret_is_always_non_negative() -> None:
    rewards = [-3.0, 5.0, 0.0, -1.0, 2.0]
    regret = LearningAnalyzer.instantaneous_regret(rewards)
    assert all(value >= 0.0 for value in regret)


def test_instantaneous_regret_of_empty_is_empty() -> None:
    assert LearningAnalyzer.instantaneous_regret([]) == []


def test_instantaneous_regret_is_zero_when_all_rewards_equal() -> None:
    assert LearningAnalyzer.instantaneous_regret([0.5, 0.5, 0.5]) == [0.0, 0.0, 0.0]


def test_cumulative_regret_is_running_sum_of_instantaneous_regret() -> None:
    rewards = [0.5, 1.0, 0.2, 1.0]
    regret = LearningAnalyzer.instantaneous_regret(rewards)
    cumulative = LearningAnalyzer.cumulative_regret(regret)
    assert cumulative == pytest.approx([0.5, 0.5, 1.3, 1.3])


def test_cumulative_regret_is_monotonically_non_decreasing() -> None:
    rewards = [3.0, -1.0, 2.0, 0.0, 5.0, -2.0]
    regret = LearningAnalyzer.instantaneous_regret(rewards)
    cumulative = LearningAnalyzer.cumulative_regret(regret)
    assert all(b >= a for a, b in zip(cumulative, cumulative[1:], strict=False))


# --- moving average ---


def test_moving_average_reward_shrinks_window_at_start() -> None:
    analyzer = LearningAnalyzer(moving_average_window=3)
    rewards = [1.0, 2.0, 3.0, 4.0]
    moving_average = analyzer.moving_average_reward(rewards)
    # window=3: [1], [1,2], [1,2,3], [2,3,4]
    assert moving_average == pytest.approx([1.0, 1.5, 2.0, 3.0])


def test_moving_average_reward_of_empty_is_empty() -> None:
    assert LearningAnalyzer().moving_average_reward([]) == []


def test_moving_average_reward_with_window_one_equals_rewards() -> None:
    rewards = [0.1, 0.9, 0.5]
    assert LearningAnalyzer().moving_average_reward(rewards, window=1) == pytest.approx(rewards)


def test_moving_average_reward_window_override_takes_precedence() -> None:
    analyzer = LearningAnalyzer(moving_average_window=10)
    rewards = [1.0, 2.0, 3.0, 4.0]
    assert analyzer.moving_average_reward(rewards, window=2) == pytest.approx(
        [1.0, 1.5, 2.5, 3.5]
    )


def test_moving_average_reward_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError, match="window"):
        LearningAnalyzer().moving_average_reward([1.0, 2.0], window=0)


def test_moving_average_reward_full_window_equals_overall_mean_at_end() -> None:
    rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
    moving_average = LearningAnalyzer(moving_average_window=5).moving_average_reward(rewards)
    assert moving_average[-1] == pytest.approx(sum(rewards) / len(rewards))


# --- convergence detection ---


def test_convergence_point_of_empty_is_none() -> None:
    assert LearningAnalyzer().convergence_point([]) is None


def test_convergence_point_of_constant_series_is_zero() -> None:
    assert LearningAnalyzer().convergence_point([0.5, 0.5, 0.5, 0.5]) == 0


def test_convergence_point_detects_late_stabilization() -> None:
    # Wanders, then settles at 1.0 from index 5 onward.
    series = [0.0, 0.8, 0.2, 0.9, 0.1, 1.0, 1.0, 1.0, 1.0]
    point = LearningAnalyzer(convergence_tolerance=0.05).convergence_point(series)
    assert point == 5


def test_convergence_point_tighter_tolerance_converges_later_or_equal() -> None:
    series = [0.0, 0.5, 0.9, 0.95, 0.99, 1.0, 1.0, 1.0]
    loose = LearningAnalyzer().convergence_point(series, tolerance=0.5)
    tight = LearningAnalyzer().convergence_point(series, tolerance=0.01)
    assert tight >= loose


def test_convergence_point_last_index_always_qualifies() -> None:
    series = [0.0, 100.0, -50.0, 3.0]  # never stabilizes early
    point = LearningAnalyzer().convergence_point(series)
    assert point == len(series) - 1


def test_convergence_point_is_included_in_analyze_metadata() -> None:
    steps = _make_steps([0.0, 0.8, 0.2, 0.9, 0.1, 1.0, 1.0, 1.0, 1.0])
    curve = LearningAnalyzer(moving_average_window=1).analyze(steps)
    assert curve.metadata["convergence_point"] == 5


# --- learning rate estimate ---


def test_learning_rate_estimate_positive_for_rising_rewards() -> None:
    rewards = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert LearningAnalyzer.learning_rate_estimate(rewards) == pytest.approx(1.0)


def test_learning_rate_estimate_negative_for_falling_rewards() -> None:
    rewards = [4.0, 3.0, 2.0, 1.0, 0.0]
    assert LearningAnalyzer.learning_rate_estimate(rewards) == pytest.approx(-1.0)


def test_learning_rate_estimate_zero_for_flat_rewards() -> None:
    assert LearningAnalyzer.learning_rate_estimate([0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.0)


def test_learning_rate_estimate_zero_for_fewer_than_two_rewards() -> None:
    assert LearningAnalyzer.learning_rate_estimate([]) == 0.0
    assert LearningAnalyzer.learning_rate_estimate([1.0]) == 0.0


# --- determinism ---


def test_analyze_is_deterministic() -> None:
    steps = _make_steps([0.1, -0.2, 0.5, 0.3, -0.1, 0.9])
    analyzer = LearningAnalyzer()

    curve_a = analyzer.analyze(steps)
    curve_b = analyzer.analyze(steps)

    assert curve_a == curve_b


def test_analyze_is_deterministic_across_independent_analyzers() -> None:
    steps = _make_steps([0.1, -0.2, 0.5, 0.3, -0.1, 0.9])

    curve_a = LearningAnalyzer().analyze(steps)
    curve_b = LearningAnalyzer().analyze(steps)

    assert curve_a == curve_b


def test_analyze_does_not_mutate_input_steps() -> None:
    steps = _make_steps([0.1, 0.2, 0.3])
    dumps_before = [step.model_dump() for step in steps]

    LearningAnalyzer().analyze(steps)

    assert [step.model_dump() for step in steps] == dumps_before
