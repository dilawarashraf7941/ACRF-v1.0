"""Unit tests for `RewardCalculator` (app/reward/calculator.py)."""

from datetime import datetime, timezone

from app.experience import ExperienceRecord
from app.reward import BaseRewardStrategy, RewardCalculator, RewardSignal, WeightedRewardStrategy


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


class _StubStrategy(BaseRewardStrategy):
    strategy_name = "StubStrategy"

    def __init__(self) -> None:
        self.received_experience: ExperienceRecord | None = None

    def compute(self, experience: ExperienceRecord) -> RewardSignal:
        self.received_experience = experience
        return RewardSignal(
            reward=42.0,
            quality_reward=0.0,
            efficiency_penalty=0.0,
            cost_penalty=0.0,
            latency_penalty=0.0,
            correction_penalty=0.0,
            completion_bonus=0.0,
            confidence=1.0,
            strategy=self.strategy_name,
            explanation="stub",
        )


def test_calculate_returns_reward_signal() -> None:
    calculator = RewardCalculator()

    signal = calculator.calculate(_make_experience())

    assert isinstance(signal, RewardSignal)


def test_defaults_to_weighted_reward_strategy() -> None:
    calculator = RewardCalculator()

    signal = calculator.calculate(_make_experience(aggregated_quality_score=0.5))

    assert signal.strategy == WeightedRewardStrategy.strategy_name


def test_uses_injected_strategy() -> None:
    stub = _StubStrategy()
    calculator = RewardCalculator(strategy=stub)

    signal = calculator.calculate(_make_experience())

    assert signal.strategy == "StubStrategy"
    assert signal.reward == 42.0


def test_passes_the_experience_through_to_the_strategy_unchanged() -> None:
    stub = _StubStrategy()
    calculator = RewardCalculator(strategy=stub)
    experience = _make_experience(session_id="session-xyz")

    calculator.calculate(experience)

    assert stub.received_experience is experience


def test_calculator_module_does_not_reference_a_repository() -> None:
    """RewardCalculator must not import or depend on any repository type."""
    import app.reward.calculator as calculator_module

    module_names = vars(calculator_module).keys()

    assert not any("repository" in name.lower() for name in module_names)


def test_is_deterministic() -> None:
    calculator = RewardCalculator()
    experience_a = _make_experience(aggregated_quality_score=0.7, iterations=1)
    experience_b = _make_experience(aggregated_quality_score=0.7, iterations=1)

    signal_a = calculator.calculate(experience_a)
    signal_b = calculator.calculate(experience_b)

    assert signal_a.model_dump(exclude={"metadata"}) == signal_b.model_dump(exclude={"metadata"})
