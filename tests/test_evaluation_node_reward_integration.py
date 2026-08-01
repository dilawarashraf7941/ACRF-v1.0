"""Integration tests for `evaluation_node`'s reward-computation behavior
(app/reward), added alongside its existing experience-recording behavior.

Only `evaluation_node` was modified to add this behavior; these tests do
not touch or assert on any other node.
"""

import pytest

from app.experience import DEFAULT_EXPERIENCE_REPOSITORY
from app.graph.nodes import evaluation_node
from app.state import AgentState, WorkerOutput


def _make_state(session_id: str = "session-1", task_id: str = "task-1") -> AgentState:
    state = AgentState(session_id=session_id, task_id=task_id, user_query="q")
    state.worker_outputs = [WorkerOutput(worker_id="worker-001", output="original response")]
    return state


def test_evaluation_node_stores_reward_in_memory_context() -> None:
    state = _make_state()

    result = evaluation_node(state)

    assert "reward" in result.memory_context


def test_reward_reflects_completion_bonus_on_success() -> None:
    state = _make_state()

    result = evaluation_node(state)

    assert result.memory_context["reward"]["completion_bonus"] > 0
    assert result.memory_context["reward"]["strategy"] == "WeightedRewardStrategy"


def test_reward_reflects_aggregated_quality_score() -> None:
    state = _make_state()
    state.aggregated_quality_score = 0.9

    result = evaluation_node(state)

    assert result.memory_context["reward"]["quality_reward"] == 0.9


def test_reward_is_attached_inside_experience_metadata() -> None:
    state = _make_state()

    result = evaluation_node(state)

    experience_metadata = result.memory_context["experience"]["metadata"]
    assert "reward" in experience_metadata
    assert experience_metadata["reward"] == result.memory_context["reward"]


def test_stored_experience_in_repository_also_has_reward_in_metadata() -> None:
    state = _make_state()

    result = evaluation_node(state)

    experience_id = result.memory_context["experience"]["experience_id"]
    stored = DEFAULT_EXPERIENCE_REPOSITORY.get(experience_id)
    assert stored is not None
    assert "reward" in stored.metadata
    assert stored.metadata["reward"]["reward"] == result.memory_context["reward"]["reward"]


def test_repository_receives_exactly_one_record_per_call() -> None:
    """The reward-less intermediate ExperienceRecord must never reach the
    repository — only the enriched (with reward in metadata) one should.
    """
    state = _make_state()

    evaluation_node(state)

    assert DEFAULT_EXPERIENCE_REPOSITORY.count() == 1


def test_reward_is_deterministic_for_identical_states() -> None:
    """`RewardCalculator` itself is a pure function of `ExperienceRecord`
    (see tests/test_reward_strategy.py and tests/test_reward_calculator.py
    for that in isolation). At the `evaluation_node` integration layer,
    `latency` is a genuine wall-clock measurement
    (`execution_metadata.updated_at - created_at`), so two independent
    real calls legitimately differ by microseconds even for identical
    logical inputs — this asserts every latency-independent component is
    exactly reproduced, and that the (tiny) latency-dependent difference
    stays within a generous tolerance.
    """
    result_a = evaluation_node(_make_state("session-a", "task-a"))
    DEFAULT_EXPERIENCE_REPOSITORY.clear()
    result_b = evaluation_node(_make_state("session-a", "task-a"))

    reward_a = result_a.memory_context["reward"]
    reward_b = result_b.memory_context["reward"]

    assert reward_a["strategy"] == reward_b["strategy"]
    assert reward_a["quality_reward"] == reward_b["quality_reward"]
    assert reward_a["cost_penalty"] == reward_b["cost_penalty"]
    assert reward_a["correction_penalty"] == reward_b["correction_penalty"]
    assert reward_a["completion_bonus"] == reward_b["completion_bonus"]
    assert reward_a["reward"] == pytest.approx(reward_b["reward"], abs=0.01)
    assert reward_a["latency_penalty"] == pytest.approx(reward_b["latency_penalty"], abs=0.01)


def test_evaluation_node_preserves_existing_memory_context_keys() -> None:
    state = _make_state()
    state.memory_context = {"existing": "value"}

    result = evaluation_node(state)

    assert result.memory_context["existing"] == "value"
    assert "reward" in result.memory_context
    assert "experience" in result.memory_context


def test_evaluation_node_still_returns_same_state_instance() -> None:
    state = _make_state()

    result = evaluation_node(state)

    assert result is state
