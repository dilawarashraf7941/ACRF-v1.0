"""Unit tests for the deterministic placeholder `evaluation_node`
(Algorithm 1, step 10: store an execution trace and finalize the
response).
"""

from app.experience import DEFAULT_EXPERIENCE_REPOSITORY
from app.graph.nodes import evaluation_node, self_correction_node
from app.state import AgentState, ExecutionStatus, WorkerOutput


def _make_state(user_query: str = "q") -> AgentState:
    state = AgentState(session_id="session-1", task_id="task-1", user_query=user_query)
    state.worker_outputs = [WorkerOutput(worker_id="worker-001", output="original response")]
    return state


def test_final_response_uses_latest_worker_output() -> None:
    state = _make_state()

    result = evaluation_node(state)

    assert result.final_response == "original response"


def test_final_response_reflects_corrected_output_when_correction_ran() -> None:
    state = _make_state()
    # A low aggregated_quality_score makes CorrectionDecisionEngine trigger
    # correction (see app/correction_policy), so self_correction_node
    # actually appends a "Placeholder corrected response." output.
    state.aggregated_quality_score = 0.1
    self_correction_node(state)

    result = evaluation_node(state)

    assert result.final_response == "Placeholder corrected response."


def test_final_response_is_empty_string_with_no_worker_outputs() -> None:
    state = AgentState(session_id="session-1", task_id="task-1", user_query="q")
    assert state.worker_outputs == []

    result = evaluation_node(state)

    assert result.final_response == ""


def test_sets_execution_status_completed() -> None:
    state = _make_state()

    result = evaluation_node(state)

    assert result.execution_status == ExecutionStatus.COMPLETED


def test_populates_evaluation_metrics() -> None:
    state = _make_state()
    state.iteration_count = 2
    state.critic_scores = {"LogicCritic": 0.0, "CodeCritic": 0.0}

    result = evaluation_node(state)

    assert result.evaluation_metrics["iteration_count"] == 2.0
    assert result.evaluation_metrics["worker_output_count"] == 1.0
    assert result.evaluation_metrics["critic_result_count"] == 2.0


def test_preserves_existing_evaluation_metrics() -> None:
    state = _make_state()
    state.evaluation_metrics = {"custom_metric": 42.0}

    result = evaluation_node(state)

    assert result.evaluation_metrics["custom_metric"] == 42.0
    assert "iteration_count" in result.evaluation_metrics


def test_execution_metadata_trace_reflects_state() -> None:
    state = _make_state()
    state.selected_critics = ["LogicCritic"]
    state.iteration_count = 1

    result = evaluation_node(state)

    trace = result.execution_metadata.metadata["trace"]
    assert trace["planner_ran"] is False
    assert trace["worker_output_count"] == 1
    assert trace["error_feature_count"] == 0
    assert trace["selected_critics"] == ["LogicCritic"]
    assert trace["iteration_count"] == 1


def test_execution_metadata_preserves_created_at_and_updates_updated_at() -> None:
    state = _make_state()
    original_created_at = state.execution_metadata.created_at
    original_updated_at = state.execution_metadata.updated_at

    result = evaluation_node(state)

    assert result.execution_metadata.created_at == original_created_at
    assert result.execution_metadata.updated_at >= original_updated_at


def test_preserves_existing_execution_metadata_extra_metadata() -> None:
    state = _make_state()
    state.execution_metadata.metadata["custom"] = "value"

    result = evaluation_node(state)

    assert result.execution_metadata.metadata["custom"] == "value"
    assert "trace" in result.execution_metadata.metadata


def test_returns_same_state_instance() -> None:
    state = _make_state()

    result = evaluation_node(state)

    assert result is state


def test_is_deterministic_apart_from_timestamp() -> None:
    result_a = evaluation_node(_make_state())
    # Same session_id/task_id/iteration_count as result_a would otherwise
    # collide on experience_id in the shared DEFAULT_EXPERIENCE_REPOSITORY
    # (see app/experience); clear between the two independent runs.
    DEFAULT_EXPERIENCE_REPOSITORY.clear()
    result_b = evaluation_node(_make_state())

    assert result_a.final_response == result_b.final_response
    assert result_a.execution_status == result_b.execution_status
    assert result_a.evaluation_metrics == result_b.evaluation_metrics
    trace_a = dict(result_a.execution_metadata.metadata["trace"])
    trace_b = dict(result_b.execution_metadata.metadata["trace"])
    assert trace_a == trace_b
