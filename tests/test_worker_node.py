"""Unit tests for the deterministic placeholder `worker_node`."""

from app.graph.nodes import planner_node, worker_node
from app.state import AgentState, ExecutionStatus, PlannerOutput, SafetyStatus


def _make_state(user_query: str) -> AgentState:
    return AgentState(session_id="session-1", task_id="task-1", user_query=user_query)


def test_worker_node_appends_worker_output() -> None:
    state = _make_state("Do something")

    result = worker_node(state)

    assert len(result.worker_outputs) == 1


def test_worker_node_populates_fixed_fields() -> None:
    state = _make_state("Do something")

    result = worker_node(state)
    output = result.worker_outputs[-1]

    assert output.worker_id == "worker-001"
    assert output.worker_name == "DefaultWorker"
    assert output.worker_type == "general"
    assert output.output == "Placeholder worker execution."
    assert output.reasoning_summary == "No reasoning performed."
    assert output.confidence == 1.0
    assert output.execution_time == 0.0
    assert output.token_usage == 0
    assert output.status == "completed"
    assert output.metadata == {}


def test_worker_node_uses_user_query_when_no_planner_output() -> None:
    state = _make_state("raw user query")
    assert state.planner_output is None

    result = worker_node(state)

    assert result.worker_outputs[-1].input == "raw user query"


def test_worker_node_uses_planner_output_original_query_when_available() -> None:
    state = _make_state("  raw   user   query  ")
    planner_node(state)

    result = worker_node(state)

    assert state.planner_output.original_query == "  raw   user   query  "
    assert result.worker_outputs[-1].input == "  raw   user   query  "


def test_worker_node_falls_back_to_user_query_when_original_query_missing() -> None:
    state = _make_state("fallback query")
    state.planner_output = PlannerOutput()  # no `original_query` extra field set

    result = worker_node(state)

    assert result.worker_outputs[-1].input == "fallback query"


def test_worker_node_falls_back_to_user_query_when_original_query_is_none() -> None:
    state = _make_state("fallback query")
    state.planner_output = PlannerOutput(original_query=None)

    result = worker_node(state)

    assert result.worker_outputs[-1].input == "fallback query"


def test_worker_node_appends_without_discarding_existing_outputs() -> None:
    state = _make_state("q")

    worker_node(state)
    worker_node(state)

    assert len(state.worker_outputs) == 2
    assert all(o.worker_id == "worker-001" for o in state.worker_outputs)


def test_worker_node_is_deterministic() -> None:
    query = "Summarize the quarterly report"

    result_a = worker_node(_make_state(query))
    result_b = worker_node(_make_state(query))

    assert result_a.worker_outputs[-1].model_dump() == result_b.worker_outputs[-1].model_dump()


def test_worker_node_returns_same_state_instance() -> None:
    state = _make_state("Any query")

    result = worker_node(state)

    assert result is state


def test_worker_node_does_not_modify_other_state_fields() -> None:
    state = _make_state("Any query")

    result = worker_node(state)

    assert result.session_id == "session-1"
    assert result.task_id == "task-1"
    assert result.user_query == "Any query"
    assert result.planner_output is None
    assert result.error_features == []
    assert result.iteration_count == 0
    assert result.max_iterations == 10
    assert result.safety_status == SafetyStatus.UNKNOWN
    assert result.execution_status == ExecutionStatus.PENDING
    assert result.final_response is None


def test_worker_node_leaves_declared_worker_output_content_field_unset() -> None:
    state = _make_state("Any query")

    result = worker_node(state)

    assert result.worker_outputs[-1].content is None
