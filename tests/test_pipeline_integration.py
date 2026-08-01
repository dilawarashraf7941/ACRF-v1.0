"""End-to-end integration tests for the ACRF pipeline, tracing Algorithm 1
(Adaptive Critic Routing): planner → worker → error feature extractor →
policy engine → router → critic → (self-correction) → evaluation.

Every node except `safety_node` is now implemented as a deterministic
placeholder (see `app/graph/nodes.py`); Algorithm 1 has no safety step, so
`safety_node` remains an unimplemented placeholder and is deliberately
never invoked by any test here. The conditional edge functions in
`app/graph/edges.py` (which decide *whether* to route to
`critic`/`self_correction`/`safety`/`evaluation`) are also still
unimplemented, so the compiled graph cannot yet branch past `router` on
its own. Per the same fallback rule used previously ("the equivalent node
sequence if the graph intentionally stops at the next unimplemented
node"), the primary integration paths here call node functions directly,
in the exact order the graph's *declared* edges specify:

    - Steps 1-6:  planner -> worker -> error_feature_extractor ->
                  policy_engine -> router                 (fixed edges)
    - Steps 7-8:  router -> critic                         (ROUTER_PATH_MAP:
                                                             "evaluate_output")
    - Step 9:     critic -> self_correction                (CRITIC_PATH_MAP:
                                                             "needs_correction")
    - Step 10:    self_correction -> evaluation             (SELF_CORRECTION_PATH_MAP:
                                                             "max_iterations_exceeded")

Note there is no declared `critic -> evaluation` edge (only `critic ->
safety` or `critic -> self_correction`) — reaching `evaluation` without
correction would require `safety_node`, which is out of scope. The
correction path above is therefore the only way to exercise the full
ten-step shape end-to-end without touching an unimplemented node, and it
is exactly Algorithm 1's "if correction required" branch.

No business logic beyond what earlier tasks in this project already
authorized is modified here — these tests only call the existing node
functions in sequence and assert on the resulting `AgentState`.
"""

import pytest

from app.experience import DEFAULT_EXPERIENCE_REPOSITORY
from app.graph.nodes import (
    NodeName,
    critic_node,
    error_feature_extractor_node,
    evaluation_node,
    planner_node,
    policy_engine_node,
    router_node,
    safety_node,
    self_correction_node,
    worker_node,
)
from app.state import AgentState, ExecutionStatus


def _run_through_router(user_query: str, task_type: str | None = None) -> AgentState:
    """Run Algorithm 1 steps 1-6: planner -> worker -> error feature
    extractor -> policy engine -> router.
    """
    state = AgentState(
        session_id="integration-session",
        task_id="integration-task",
        user_query=user_query,
        task_type=task_type,
    )
    state = planner_node(state)
    state = worker_node(state)
    state = error_feature_extractor_node(state)
    state = policy_engine_node(state)
    state = router_node(state)
    return state


def _run_through_critic(user_query: str, task_type: str | None = None) -> AgentState:
    """Run Algorithm 1 steps 1-8: `_run_through_router` plus critic execution/aggregation."""
    state = _run_through_router(user_query, task_type=task_type)
    return critic_node(state)


def _run_full_pipeline_via_correction(user_query: str, task_type: str | None = None) -> AgentState:
    """Run Algorithm 1 steps 1-10 via the "correction required" branch:
    `_run_through_critic` plus self-correction and final evaluation.
    """
    state = _run_through_critic(user_query, task_type=task_type)
    state = self_correction_node(state)
    return evaluation_node(state)


# --- Per-stage "Verify" assertions ---


def test_planner_populated() -> None:
    state = _run_through_router("Explain how binary search works")

    assert state.planner_output is not None
    assert state.planner_output.original_query == "Explain how binary search works"
    assert state.planner_output.task_type == "general"


def test_worker_populated() -> None:
    state = _run_through_router("Explain how binary search works")

    assert len(state.worker_outputs) == 1
    assert state.worker_outputs[0].worker_id == "worker-001"
    assert state.worker_outputs[0].status == "completed"


def test_error_features_extracted() -> None:
    state = _run_through_router("Explain how binary search works")

    assert len(state.error_features) == 1
    assert state.error_features[0].source_node == NodeName.ERROR_FEATURE_EXTRACTOR.value


def test_policy_engine_records_candidate_scoring() -> None:
    state = _run_through_router("Explain how binary search works")

    policy_engine_entry = state.memory_context["policy_engine"]
    candidate_critics = policy_engine_entry["candidate_critics"]
    assert len(candidate_critics) == 4
    assert set(policy_engine_entry["scores"].keys()) == set(candidate_critics)
    assert any(score != 0.0 for score in policy_engine_entry["scores"].values())


def test_critics_selected() -> None:
    state = _run_through_router("Explain how binary search works")

    assert state.selected_critics == ["LogicCritic"]


def test_critics_selected_for_code_task_type() -> None:
    state = _run_through_router("Write a function", task_type="code")

    assert state.selected_critics == ["CodeCritic"]


def test_policy_decision_exists() -> None:
    state = _run_through_router("Explain how binary search works")

    assert state.policy_decision is not None
    assert state.policy_decision.action == "select_critics"
    assert state.policy_decision.target_node == NodeName.CRITIC.value


def test_policy_engine_diagnostics_do_not_override_router_selection() -> None:
    """`policy_engine_node`'s heuristic scoring never reads `task_type` (it
    is not in its declared input set), so for a non-code task its own
    diagnostic top pick can legitimately differ from `router_node`'s
    task_type-driven authoritative selection; confirm `router_node`'s
    output is what actually lands in `selected_critics`.
    """
    state = _run_through_router("Summarize this", task_type="research")

    policy_engine_pick = state.memory_context["policy_engine"]["selected_critics"]
    assert policy_engine_pick != state.selected_critics
    assert state.selected_critics == ["LogicCritic"]


# --- Critic execution and aggregation (steps 7-8) ---


def test_critic_results_populated() -> None:
    state = _run_through_critic("Explain how binary search works")

    assert state.critic_scores == {"LogicCritic": 0.0}
    assert len(state.critic_feedback) == 1
    assert state.aggregated_quality_score == 0.0


def test_critic_executes_exactly_the_critics_router_selected() -> None:
    state = _run_through_critic("Write a function", task_type="code")

    assert set(state.critic_scores.keys()) == set(state.selected_critics) == {"CodeCritic"}


# --- Full ten-step path via the correction branch (steps 9-10) ---


def test_full_pipeline_via_correction_reaches_terminal_state() -> None:
    state = _run_full_pipeline_via_correction("Explain how binary search works")

    assert state.execution_status == ExecutionStatus.COMPLETED
    assert state.final_response == "Placeholder corrected response."
    assert len(state.correction_history) == 1
    assert state.iteration_count == 1
    assert state.evaluation_metrics["worker_output_count"] == 2.0
    assert state.execution_metadata.metadata["trace"]["iteration_count"] == 1


# --- safety_node is out of scope for Algorithm 1 and remains unimplemented ---


def test_safety_node_remains_unimplemented_and_is_never_invoked() -> None:
    state = _run_full_pipeline_via_correction("Explain how binary search works")

    with pytest.raises(NotImplementedError):
        safety_node(state)


# --- Determinism across repeated runs ---


def test_pipeline_through_critic_is_deterministic() -> None:
    result_a = _run_through_critic("Summarize the quarterly report", task_type="research")
    result_b = _run_through_critic("Summarize the quarterly report", task_type="research")

    assert result_a.selected_critics == result_b.selected_critics
    assert result_a.critic_scores == result_b.critic_scores
    assert result_a.aggregated_quality_score == result_b.aggregated_quality_score


def test_full_pipeline_via_correction_is_deterministic() -> None:
    result_a = _run_full_pipeline_via_correction("Summarize the quarterly report")
    # Both runs use the same hardcoded session_id/task_id, which would
    # otherwise collide on experience_id in the shared
    # DEFAULT_EXPERIENCE_REPOSITORY (see app/experience); clear between
    # the two independent runs.
    DEFAULT_EXPERIENCE_REPOSITORY.clear()
    result_b = _run_full_pipeline_via_correction("Summarize the quarterly report")

    assert result_a.final_response == result_b.final_response
    assert result_a.execution_status == result_b.execution_status
    assert result_a.iteration_count == result_b.iteration_count


# --- The real compiled LangGraph graph: confirms wiring and the true stop point ---


def test_compiled_graph_runs_implemented_nodes_then_stops_at_unimplemented_routing() -> None:
    """Invoke the actual compiled graph (not a hand-rolled node sequence).

    The graph's fixed edges run `planner -> worker ->
    error_feature_extractor -> policy_engine -> router`; all five are now
    implemented. Immediately after `router`, the graph must evaluate the
    conditional edge function `route_after_router` (see
    `app/graph/edges.py`) to decide the next node — and that function is
    still an unimplemented placeholder. So invoking the compiled graph
    must execute the full five-node prefix for real, then raise
    `NotImplementedError` from `route_after_router`, never reaching
    `critic` through the graph itself.
    """
    from app.graph.state_graph import compile_graph

    state = AgentState(
        session_id="integration-session-graph",
        task_id="integration-task-graph",
        user_query="Explain how binary search works",
    )
    compiled = compile_graph()

    with pytest.raises(NotImplementedError, match="route_after_router"):
        compiled.invoke(state)
