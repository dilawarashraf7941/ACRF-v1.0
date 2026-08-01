"""Unit tests for `OfflineEvaluator` (`app/evaluation/offline/evaluator.py`)."""

from datetime import datetime, timezone

import pytest

from app.context import ContextVector
from app.evaluation.offline.evaluator import OfflineEvaluator
from app.evaluation.offline.models import ReplayResult
from app.evaluation.offline.replay import ReplayEngine
from app.experience import ExperienceRecord, InMemoryExperienceRepository
from app.reward import RewardCalculator

_TIMESTAMP = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _make_experience(
    experience_id: str,
    selected_critics: list[str],
    quality: float | None,
    iterations: int,
    latency: float | None,
) -> ExperienceRecord:
    return ExperienceRecord(
        experience_id=experience_id,
        session_id="session-1",
        task_id="task-1",
        timestamp=_TIMESTAMP,
        selected_critics=selected_critics,
        critic_scores={critic: 0.7 for critic in selected_critics},
        aggregated_quality_score=quality,
        iterations=iterations,
        execution_status="completed",
        latency=latency,
    )


class _AlwaysSelectPolicy:
    def __init__(self, critic: str) -> None:
        self._critic = critic

    def select_action(self, context: ContextVector, candidate_actions: list[str]) -> object:
        class _Decision:
            selected_critics = [self._critic]

        return _Decision()


def _make_engine(
    repository: InMemoryExperienceRepository, critic: str = "CodeCritic"
) -> ReplayEngine:
    return ReplayEngine(
        repository=repository,
        policy=_AlwaysSelectPolicy(critic),
        reward_calculator=RewardCalculator(),
    )


def test_evaluate_empty_repository_returns_zeroed_result() -> None:
    repository = InMemoryExperienceRepository()
    engine = _make_engine(repository)

    result = OfflineEvaluator().evaluate(engine, policy_name="HeuristicPolicy")

    assert isinstance(result, ReplayResult)
    assert result.policy_name == "HeuristicPolicy"
    assert result.total_experiences == 0
    assert result.total_reward == 0.0
    assert result.average_reward == 0.0
    assert result.average_quality == 0.0
    assert result.average_iterations == 0.0
    assert result.average_latency == 0.0
    assert result.critic_selection_frequency == {}


def test_evaluate_records_total_experiences_as_matched_count() -> None:
    repository = InMemoryExperienceRepository()
    repository.add(_make_experience("exp-1", ["CodeCritic"], 0.8, 1, 1.0))
    repository.add(_make_experience("exp-2", ["LogicCritic"], 0.5, 2, 2.0))
    repository.add(_make_experience("exp-3", ["CodeCritic"], 0.6, 3, 3.0))
    engine = _make_engine(repository)

    result = OfflineEvaluator().evaluate(engine, policy_name="HeuristicPolicy")

    assert result.total_experiences == 2
    assert result.metadata["total_stored_experiences"] == 3
    assert result.metadata["match_rate"] == pytest.approx(2 / 3)


def test_evaluate_computes_average_reward_correctly() -> None:
    repository = InMemoryExperienceRepository()
    repository.add(_make_experience("exp-1", ["CodeCritic"], 1.0, 0, 0.0))
    repository.add(_make_experience("exp-2", ["CodeCritic"], 0.0, 0, 0.0))
    engine = _make_engine(repository)
    reward_calculator = RewardCalculator()

    result = OfflineEvaluator().evaluate(engine, policy_name="HeuristicPolicy")

    expected_rewards = [
        reward_calculator.calculate(experience).reward for experience in repository.list()
    ]
    assert result.total_reward == pytest.approx(sum(expected_rewards))
    assert result.average_reward == pytest.approx(sum(expected_rewards) / 2)


def test_evaluate_computes_average_quality_treating_missing_as_zero() -> None:
    repository = InMemoryExperienceRepository()
    repository.add(_make_experience("exp-1", ["CodeCritic"], 1.0, 0, 0.0))
    repository.add(_make_experience("exp-2", ["CodeCritic"], None, 0, 0.0))
    engine = _make_engine(repository)

    result = OfflineEvaluator().evaluate(engine, policy_name="HeuristicPolicy")

    assert result.average_quality == pytest.approx(0.5)


def test_evaluate_computes_average_latency_treating_missing_as_zero() -> None:
    repository = InMemoryExperienceRepository()
    repository.add(_make_experience("exp-1", ["CodeCritic"], 0.5, 0, 2.0))
    repository.add(_make_experience("exp-2", ["CodeCritic"], 0.5, 0, None))
    engine = _make_engine(repository)

    result = OfflineEvaluator().evaluate(engine, policy_name="HeuristicPolicy")

    assert result.average_latency == pytest.approx(1.0)


def test_evaluate_computes_average_iterations() -> None:
    repository = InMemoryExperienceRepository()
    repository.add(_make_experience("exp-1", ["CodeCritic"], 0.5, 2, 0.0))
    repository.add(_make_experience("exp-2", ["CodeCritic"], 0.5, 4, 0.0))
    engine = _make_engine(repository)

    result = OfflineEvaluator().evaluate(engine, policy_name="HeuristicPolicy")

    assert result.average_iterations == pytest.approx(3.0)


def test_evaluate_computes_critic_selection_frequency() -> None:
    repository = InMemoryExperienceRepository()
    repository.add(_make_experience("exp-1", ["CodeCritic"], 0.5, 0, 0.0))
    repository.add(_make_experience("exp-2", ["CodeCritic"], 0.5, 0, 0.0))
    repository.add(_make_experience("exp-3", ["LogicCritic"], 0.5, 0, 0.0))
    engine = _make_engine(repository)  # always selects "CodeCritic"

    result = OfflineEvaluator().evaluate(engine, policy_name="HeuristicPolicy")

    assert result.total_experiences == 2
    assert result.critic_selection_frequency == {"CodeCritic": 1.0}


def test_evaluate_is_deterministic() -> None:
    repository = InMemoryExperienceRepository()
    repository.add(_make_experience("exp-1", ["CodeCritic"], 0.7, 1, 1.5))
    repository.add(_make_experience("exp-2", ["LogicCritic"], 0.3, 2, 2.5))
    evaluator = OfflineEvaluator()

    result_a = evaluator.evaluate(_make_engine(repository), policy_name="HeuristicPolicy")
    result_b = evaluator.evaluate(_make_engine(repository), policy_name="HeuristicPolicy")

    assert result_a == result_b


def test_evaluate_uses_the_given_policy_name_independent_of_policy_object() -> None:
    repository = InMemoryExperienceRepository()
    engine = _make_engine(repository)

    result = OfflineEvaluator().evaluate(engine, policy_name="CustomLabel")

    assert result.policy_name == "CustomLabel"
