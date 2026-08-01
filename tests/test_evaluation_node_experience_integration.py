"""Integration tests for `evaluation_node`'s experience-recording behavior
(app/experience), added alongside its existing responsibilities.

Only `evaluation_node` was modified to add this behavior; these tests do
not touch or assert on any other node.
"""

from app.experience import DEFAULT_EXPERIENCE_REPOSITORY
from app.graph.nodes import evaluation_node
from app.state import AgentState, WorkerOutput


def _make_state(session_id: str = "session-1", task_id: str = "task-1") -> AgentState:
    state = AgentState(session_id=session_id, task_id=task_id, user_query="q")
    state.worker_outputs = [WorkerOutput(worker_id="worker-001", output="original response")]
    return state


def test_evaluation_node_stores_experience_in_memory_context() -> None:
    state = _make_state()

    result = evaluation_node(state)

    assert "experience" in result.memory_context


def test_experience_in_memory_context_matches_final_state() -> None:
    state = _make_state()
    state.selected_critics = ["LogicCritic"]
    state.critic_scores = {"LogicCritic": 0.4}
    state.aggregated_quality_score = 0.4

    result = evaluation_node(state)

    experience = result.memory_context["experience"]
    assert experience["session_id"] == "session-1"
    assert experience["task_id"] == "task-1"
    assert experience["final_response"] == result.final_response
    assert experience["execution_status"] == result.execution_status.value
    assert experience["selected_critics"] == ["LogicCritic"]
    assert experience["critic_scores"] == {"LogicCritic": 0.4}
    assert experience["aggregated_quality_score"] == 0.4
    assert experience["iterations"] == result.iteration_count


def test_evaluation_node_stores_experience_in_default_repository() -> None:
    state = _make_state()

    result = evaluation_node(state)

    experience_id = result.memory_context["experience"]["experience_id"]
    stored = DEFAULT_EXPERIENCE_REPOSITORY.get(experience_id)
    assert stored is not None
    assert stored.session_id == "session-1"


def test_evaluation_node_experience_id_is_deterministic() -> None:
    result_a = evaluation_node(_make_state("session-a", "task-a"))
    DEFAULT_EXPERIENCE_REPOSITORY.clear()
    result_b = evaluation_node(_make_state("session-a", "task-a"))

    assert (
        result_a.memory_context["experience"]["experience_id"]
        == result_b.memory_context["experience"]["experience_id"]
    )


def test_evaluation_node_experience_id_differs_for_different_sessions() -> None:
    result_a = evaluation_node(_make_state("session-a", "task-a"))
    result_b = evaluation_node(_make_state("session-b", "task-a"))

    assert (
        result_a.memory_context["experience"]["experience_id"]
        != result_b.memory_context["experience"]["experience_id"]
    )


def test_evaluation_node_still_sets_final_response_and_status() -> None:
    """The pre-existing evaluation_node behavior is unaffected by adding
    experience recording.
    """
    from app.state import ExecutionStatus

    state = _make_state()

    result = evaluation_node(state)

    assert result.final_response == "original response"
    assert result.execution_status == ExecutionStatus.COMPLETED


def test_evaluation_node_preserves_existing_memory_context_keys() -> None:
    state = _make_state()
    state.memory_context = {"existing": "value"}

    result = evaluation_node(state)

    assert result.memory_context["existing"] == "value"
    assert "experience" in result.memory_context


def test_evaluation_node_returns_same_state_instance() -> None:
    state = _make_state()

    result = evaluation_node(state)

    assert result is state
