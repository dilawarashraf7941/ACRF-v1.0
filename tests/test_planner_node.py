"""Unit tests for the deterministic placeholder `planner_node`."""

from app.graph.nodes import planner_node
from app.state import AgentState, ExecutionStatus, SafetyStatus


def _make_state(user_query: str) -> AgentState:
    return AgentState(session_id="session-1", task_id="task-1", user_query=user_query)


def test_planner_node_populates_planner_output() -> None:
    state = _make_state("What is ACRF?")

    result = planner_node(state)

    assert result.planner_output is not None
    assert result.planner_output.original_query == "What is ACRF?"
    assert result.planner_output.normalized_query == "What is ACRF?"
    assert result.planner_output.task_type == "general"
    assert result.planner_output.decomposition == []
    assert result.planner_output.planning_notes == "Placeholder planner"


def test_planner_node_normalizes_whitespace() -> None:
    state = _make_state("  What   is\tACRF?  \n")

    result = planner_node(state)

    assert result.planner_output.original_query == "  What   is\tACRF?  \n"
    assert result.planner_output.normalized_query == "What is ACRF?"


def test_planner_node_handles_empty_query() -> None:
    state = _make_state("")

    result = planner_node(state)

    assert result.planner_output.original_query == ""
    assert result.planner_output.normalized_query == ""


def test_planner_node_is_deterministic() -> None:
    query = "Summarize the quarterly report"

    result_a = planner_node(_make_state(query))
    result_b = planner_node(_make_state(query))

    assert result_a.planner_output.model_dump() == result_b.planner_output.model_dump()


def test_planner_node_returns_same_state_instance() -> None:
    state = _make_state("Any query")

    result = planner_node(state)

    assert result is state


def test_planner_node_does_not_modify_other_state_fields() -> None:
    state = _make_state("Any query")

    result = planner_node(state)

    assert result.session_id == "session-1"
    assert result.task_id == "task-1"
    assert result.user_query == "Any query"
    assert result.worker_outputs == []
    assert result.error_features == []
    assert result.iteration_count == 0
    assert result.max_iterations == 10
    assert result.safety_status == SafetyStatus.UNKNOWN
    assert result.execution_status == ExecutionStatus.PENDING
    assert result.final_response is None


def test_planner_node_leaves_declared_planner_output_fields_at_defaults() -> None:
    state = _make_state("Any query")

    result = planner_node(state)

    assert result.planner_output.summary is None
    assert result.planner_output.steps == []
    assert result.planner_output.metadata == {}
