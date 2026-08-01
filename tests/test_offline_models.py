"""Unit tests for `ReplayStep`, `ReplayResult`, `BenchmarkResult`.

See `app/evaluation/offline/models.py`.
"""

import pytest
from pydantic import ValidationError

from app.evaluation.offline.models import BenchmarkResult, ReplayResult, ReplayStep


def _make_replay_step(**overrides: object) -> ReplayStep:
    defaults: dict[str, object] = {
        "experience_id": "exp-1",
        "context_id": "ctx-1",
        "selected_critics": ["CodeCritic"],
        "reward": 0.5,
        "iterations": 1,
    }
    defaults.update(overrides)
    return ReplayStep(**defaults)


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


# --- ReplayStep ---


def test_replay_step_constructs_with_required_fields() -> None:
    step = _make_replay_step()
    assert step.experience_id == "exp-1"
    assert step.quality is None
    assert step.latency is None


def test_replay_step_missing_required_field_raises() -> None:
    with pytest.raises(ValidationError):
        ReplayStep(context_id="ctx-1", selected_critics=[], reward=0.5, iterations=1)


def test_replay_step_negative_iterations_raises() -> None:
    with pytest.raises(ValidationError):
        _make_replay_step(iterations=-1)


def test_replay_step_is_frozen() -> None:
    step = _make_replay_step()
    with pytest.raises(ValidationError):
        step.reward = 1.0  # type: ignore[misc]


# --- ReplayResult ---


def test_replay_result_constructs_with_required_fields() -> None:
    result = _make_replay_result()
    assert result.policy_name == "HeuristicPolicy"
    assert result.critic_selection_frequency == {}
    assert result.metadata == {}


def test_replay_result_negative_total_experiences_raises() -> None:
    with pytest.raises(ValidationError):
        _make_replay_result(total_experiences=-1)


def test_replay_result_holds_critic_selection_frequency() -> None:
    result = _make_replay_result(critic_selection_frequency={"CodeCritic": 1.0})
    assert result.critic_selection_frequency == {"CodeCritic": 1.0}


def test_replay_result_is_frozen() -> None:
    result = _make_replay_result()
    with pytest.raises(ValidationError):
        result.average_reward = 1.0  # type: ignore[misc]


def test_replay_result_missing_required_field_raises() -> None:
    with pytest.raises(ValidationError):
        ReplayResult(policy_name="HeuristicPolicy")


# --- BenchmarkResult ---


def test_benchmark_result_constructs_with_required_fields() -> None:
    result = BenchmarkResult(
        baseline_policy="HeuristicPolicy",
        candidate_policy="LinUCBPolicy",
        reward_improvement=0.1,
        quality_improvement=0.05,
        latency_difference=-0.2,
        iteration_difference=-1.0,
        winner="LinUCBPolicy",
    )
    assert result.winner == "LinUCBPolicy"
    assert result.metadata == {}


def test_benchmark_result_is_frozen() -> None:
    result = BenchmarkResult(
        baseline_policy="HeuristicPolicy",
        candidate_policy="LinUCBPolicy",
        reward_improvement=0.1,
        quality_improvement=0.05,
        latency_difference=-0.2,
        iteration_difference=-1.0,
        winner="tie",
    )
    with pytest.raises(ValidationError):
        result.winner = "HeuristicPolicy"  # type: ignore[misc]


def test_benchmark_result_missing_required_field_raises() -> None:
    with pytest.raises(ValidationError):
        BenchmarkResult(baseline_policy="HeuristicPolicy", candidate_policy="LinUCBPolicy")
