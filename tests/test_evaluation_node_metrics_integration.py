"""Integration tests for `evaluation_node`'s metrics-collection behavior
(app/metrics), added alongside its existing experience/reward behavior.

Only `evaluation_node` was modified to add this behavior; these tests do
not touch or assert on any other node.
"""

from app.graph.nodes import evaluation_node
from app.metrics import DEFAULT_METRICS_REPOSITORY
from app.state import AgentState, WorkerOutput


def _make_state(session_id: str = "session-1", task_id: str = "task-1") -> AgentState:
    state = AgentState(session_id=session_id, task_id=task_id, user_query="q")
    state.worker_outputs = [WorkerOutput(worker_id="worker-001", output="original response")]
    return state


def test_evaluation_node_stores_metrics_in_memory_context() -> None:
    state = _make_state()

    result = evaluation_node(state)

    assert "metrics" in result.memory_context


def test_metrics_in_memory_context_matches_final_state() -> None:
    state = _make_state()
    state.aggregated_quality_score = 0.6
    state.selected_critics = ["LogicCritic"]

    result = evaluation_node(state)

    metrics = result.memory_context["metrics"]
    assert metrics["execution_id"] == result.memory_context["experience"]["experience_id"]
    assert metrics["reward"] == result.memory_context["reward"]["reward"]
    assert metrics["aggregated_quality_score"] == 0.6
    assert metrics["selected_critics"] == ["LogicCritic"]
    assert metrics["execution_status"] == result.execution_status.value
    assert metrics["iterations"] == result.iteration_count


def test_evaluation_node_stores_metrics_in_default_repository() -> None:
    state = _make_state()

    result = evaluation_node(state)

    assert DEFAULT_METRICS_REPOSITORY.count() == 1
    stored = DEFAULT_METRICS_REPOSITORY.list()[0]
    assert stored.execution_id == result.memory_context["metrics"]["execution_id"]


def test_metrics_summary_reflects_stored_run() -> None:
    state = _make_state()
    state.aggregated_quality_score = 1.0

    evaluation_node(state)

    summary = DEFAULT_METRICS_REPOSITORY.summary()
    assert summary.total_runs == 1
    assert summary.success_rate == 1.0


def test_metrics_correction_applied_false_without_correction() -> None:
    state = _make_state()

    result = evaluation_node(state)

    assert result.memory_context["metrics"]["correction_applied"] is False


def test_evaluation_node_preserves_existing_memory_context_keys() -> None:
    state = _make_state()
    state.memory_context = {"existing": "value"}

    result = evaluation_node(state)

    assert result.memory_context["existing"] == "value"
    assert "metrics" in result.memory_context
    assert "experience" in result.memory_context
    assert "reward" in result.memory_context


def test_evaluation_node_still_returns_same_state_instance() -> None:
    state = _make_state()

    result = evaluation_node(state)

    assert result is state


def test_multiple_runs_accumulate_in_repository() -> None:
    evaluation_node(_make_state("session-a", "task-a"))
    evaluation_node(_make_state("session-b", "task-b"))

    assert DEFAULT_METRICS_REPOSITORY.count() == 2
    assert DEFAULT_METRICS_REPOSITORY.summary().total_runs == 2
