"""Unit tests for `WeightedRewardStrategy` and `BaseRewardStrategy`
(app/reward/strategy.py).
"""

from datetime import datetime, timezone

import pytest

from app.experience import ExperienceRecord
from app.reward import BaseRewardStrategy, RewardSignal, WeightedRewardStrategy
from app.reward.strategy import (
    COMPLETION_BONUS,
    FAILURE_PENALTY,
    MAX_CORRECTION_PENALTY,
    MAX_COST_PENALTY,
    MAX_LATENCY_PENALTY,
)


def _make_experience(**overrides: object) -> ExperienceRecord:
    defaults: dict[str, object] = {
        "experience_id": "id-1",
        "session_id": "s1",
        "task_id": "t1",
        "timestamp": datetime.now(timezone.utc),
        "iterations": 0,
        "execution_status": "completed",
    }
    defaults.update(overrides)
    return ExperienceRecord(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def strategy() -> WeightedRewardStrategy:
    return WeightedRewardStrategy()


# --- Basic contract ---


def test_compute_returns_reward_signal(strategy: WeightedRewardStrategy) -> None:
    signal = strategy.compute(_make_experience())

    assert isinstance(signal, RewardSignal)


def test_strategy_name_is_recorded(strategy: WeightedRewardStrategy) -> None:
    signal = strategy.compute(_make_experience())

    assert signal.strategy == "WeightedRewardStrategy"


def test_is_a_base_reward_strategy(strategy: WeightedRewardStrategy) -> None:
    assert isinstance(strategy, BaseRewardStrategy)


def test_base_reward_strategy_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        BaseRewardStrategy()  # type: ignore[abstract]


# --- Higher quality -> higher reward ---


def test_higher_quality_produces_higher_reward(strategy: WeightedRewardStrategy) -> None:
    low = strategy.compute(_make_experience(aggregated_quality_score=0.1))
    high = strategy.compute(_make_experience(aggregated_quality_score=0.9))

    assert high.reward > low.reward
    assert high.quality_reward > low.quality_reward


def test_quality_reward_is_clamped_to_zero_one(strategy: WeightedRewardStrategy) -> None:
    over = strategy.compute(_make_experience(aggregated_quality_score=5.0))
    under = strategy.compute(_make_experience(aggregated_quality_score=-5.0))

    assert over.quality_reward == 1.0
    assert under.quality_reward == 0.0


def test_missing_quality_score_contributes_zero(strategy: WeightedRewardStrategy) -> None:
    signal = strategy.compute(_make_experience(aggregated_quality_score=None))

    assert signal.quality_reward == 0.0


# --- More iterations -> lower reward (correction_penalty) ---


def test_more_iterations_reduce_reward(strategy: WeightedRewardStrategy) -> None:
    few = strategy.compute(_make_experience(iterations=0))
    many = strategy.compute(_make_experience(iterations=5))

    assert many.reward < few.reward
    assert many.correction_penalty > few.correction_penalty


def test_correction_penalty_is_bounded(strategy: WeightedRewardStrategy) -> None:
    signal = strategy.compute(_make_experience(iterations=1000))

    assert signal.correction_penalty == MAX_CORRECTION_PENALTY


def test_zero_iterations_yields_zero_correction_penalty(strategy: WeightedRewardStrategy) -> None:
    signal = strategy.compute(_make_experience(iterations=0))

    assert signal.correction_penalty == 0.0


# --- Higher cost -> lower reward ---


def test_higher_cost_reduces_reward(strategy: WeightedRewardStrategy) -> None:
    cheap = strategy.compute(_make_experience(estimated_cost=0.0))
    expensive = strategy.compute(_make_experience(estimated_cost=100.0))

    assert expensive.reward < cheap.reward
    assert expensive.cost_penalty > cheap.cost_penalty


def test_cost_penalty_is_bounded(strategy: WeightedRewardStrategy) -> None:
    signal = strategy.compute(_make_experience(estimated_cost=1_000_000.0))

    assert signal.cost_penalty == MAX_COST_PENALTY


def test_missing_cost_contributes_zero_penalty(strategy: WeightedRewardStrategy) -> None:
    signal = strategy.compute(_make_experience(estimated_cost=None))

    assert signal.cost_penalty == 0.0


def test_negative_cost_does_not_become_a_bonus(strategy: WeightedRewardStrategy) -> None:
    signal = strategy.compute(_make_experience(estimated_cost=-50.0))

    assert signal.cost_penalty == 0.0


# --- Latency penalty ---


def test_higher_latency_reduces_reward(strategy: WeightedRewardStrategy) -> None:
    fast = strategy.compute(_make_experience(latency=0.0))
    slow = strategy.compute(_make_experience(latency=30.0))

    assert slow.reward < fast.reward
    assert slow.latency_penalty > fast.latency_penalty


def test_latency_penalty_is_bounded(strategy: WeightedRewardStrategy) -> None:
    signal = strategy.compute(_make_experience(latency=100_000.0))

    assert signal.latency_penalty == MAX_LATENCY_PENALTY


def test_missing_latency_contributes_zero_penalty(strategy: WeightedRewardStrategy) -> None:
    signal = strategy.compute(_make_experience(latency=None))

    assert signal.latency_penalty == 0.0


# --- Completion bonus / failure adjustment ---


def test_completed_execution_gets_bonus(strategy: WeightedRewardStrategy) -> None:
    signal = strategy.compute(_make_experience(execution_status="completed"))

    assert signal.completion_bonus == COMPLETION_BONUS


def test_failed_execution_gets_negative_adjustment(strategy: WeightedRewardStrategy) -> None:
    signal = strategy.compute(_make_experience(execution_status="failed"))

    assert signal.completion_bonus == -FAILURE_PENALTY
    assert signal.completion_bonus < 0


def test_completed_reward_exceeds_failed_reward_all_else_equal(
    strategy: WeightedRewardStrategy,
) -> None:
    completed = strategy.compute(
        _make_experience(execution_status="completed", aggregated_quality_score=0.5)
    )
    failed = strategy.compute(
        _make_experience(execution_status="failed", aggregated_quality_score=0.5)
    )

    assert completed.reward > failed.reward


def test_unknown_execution_status_is_neutral(strategy: WeightedRewardStrategy) -> None:
    signal = strategy.compute(_make_experience(execution_status="some_future_status"))

    assert signal.completion_bonus == 0.0


# --- efficiency_penalty rollup ---


def test_efficiency_penalty_equals_cost_plus_latency(strategy: WeightedRewardStrategy) -> None:
    signal = strategy.compute(_make_experience(estimated_cost=10.0, latency=2.0))

    assert signal.efficiency_penalty == pytest.approx(signal.cost_penalty + signal.latency_penalty)


def test_efficiency_penalty_not_double_subtracted_from_reward(
    strategy: WeightedRewardStrategy,
) -> None:
    experience = _make_experience(
        aggregated_quality_score=0.5, estimated_cost=10.0, latency=2.0, iterations=1
    )
    signal = strategy.compute(experience)

    expected_reward = (
        signal.quality_reward
        + signal.completion_bonus
        - signal.cost_penalty
        - signal.latency_penalty
        - signal.correction_penalty
    )
    assert signal.reward == pytest.approx(expected_reward)


# --- Graceful handling of missing/future fields ---


def test_experience_with_no_optional_fields_produces_valid_signal(
    strategy: WeightedRewardStrategy,
) -> None:
    signal = strategy.compute(_make_experience())

    assert signal.reward == COMPLETION_BONUS
    assert signal.quality_reward == 0.0
    assert signal.cost_penalty == 0.0
    assert signal.latency_penalty == 0.0
    assert signal.correction_penalty == 0.0


def test_confidence_is_zero_when_all_optional_signals_missing(
    strategy: WeightedRewardStrategy,
) -> None:
    signal = strategy.compute(_make_experience())

    assert signal.confidence == 0.0


def test_confidence_is_one_when_all_optional_signals_present(
    strategy: WeightedRewardStrategy,
) -> None:
    signal = strategy.compute(
        _make_experience(aggregated_quality_score=0.5, estimated_cost=1.0, latency=1.0)
    )

    assert signal.confidence == 1.0


def test_confidence_is_partial_with_some_signals_present(strategy: WeightedRewardStrategy) -> None:
    signal = strategy.compute(_make_experience(aggregated_quality_score=0.5))

    assert signal.confidence == pytest.approx(1 / 3)


def test_extra_experience_fields_do_not_break_computation(strategy: WeightedRewardStrategy) -> None:
    """Future ExperienceRecord fields (via extra='allow') must not crash the strategy."""
    experience = _make_experience(some_future_field="unexpected value")

    signal = strategy.compute(experience)

    assert isinstance(signal, RewardSignal)


# --- Deterministic calculations ---


def test_is_deterministic_for_identical_experiences(strategy: WeightedRewardStrategy) -> None:
    experience_a = _make_experience(aggregated_quality_score=0.6, iterations=2, estimated_cost=3.0)
    experience_b = _make_experience(aggregated_quality_score=0.6, iterations=2, estimated_cost=3.0)

    signal_a = strategy.compute(experience_a)
    signal_b = strategy.compute(experience_b)

    assert signal_a.model_dump(exclude={"metadata"}) == signal_b.model_dump(exclude={"metadata"})


def test_metadata_records_inputs_and_weights(strategy: WeightedRewardStrategy) -> None:
    experience = _make_experience(aggregated_quality_score=0.5)

    signal = strategy.compute(experience)

    assert signal.metadata["experience_id"] == "id-1"
    assert signal.metadata["inputs"]["aggregated_quality_score"] == 0.5
    assert "quality_reward_weight" in signal.metadata["weights"]
