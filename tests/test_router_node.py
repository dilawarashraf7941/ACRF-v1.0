"""Unit tests for the deterministic, rule-based placeholder `router_node`."""

from app.graph.nodes import NodeName, planner_node, router_node, worker_node
from app.state import AgentState, ExecutionStatus, PlannerOutput, SafetyStatus


def _make_state(user_query: str = "q", task_type: str | None = None) -> AgentState:
    return AgentState(
        session_id="session-1", task_id="task-1", user_query=user_query, task_type=task_type
    )


def test_router_node_selects_code_critic_for_code_task_type() -> None:
    state = _make_state(task_type="code")

    result = router_node(state)

    assert result.selected_critics == ["CodeCritic"]


def test_router_node_selects_logic_critic_for_non_code_task_type() -> None:
    state = _make_state(task_type="research")

    result = router_node(state)

    assert result.selected_critics == ["LogicCritic"]


def test_router_node_selects_logic_critic_when_task_type_unset() -> None:
    state = _make_state(task_type=None)
    assert state.planner_output is None

    result = router_node(state)

    assert result.selected_critics == ["LogicCritic"]


def test_router_node_falls_back_to_planner_output_task_type() -> None:
    state = _make_state(task_type=None)
    state.planner_output = PlannerOutput(task_type="code")

    result = router_node(state)

    assert result.selected_critics == ["CodeCritic"]


def test_router_node_prefers_state_task_type_over_planner_output() -> None:
    state = _make_state(task_type="code")
    state.planner_output = PlannerOutput(task_type="research")

    result = router_node(state)

    assert result.selected_critics == ["CodeCritic"]


def test_router_node_uses_planner_node_output_task_type_as_fallback() -> None:
    # planner_node always sets planner_output.task_type = "general",
    # which does not match the "code" rule and should route to LogicCritic.
    state = _make_state(task_type=None)
    planner_node(state)

    result = router_node(state)

    assert state.planner_output.task_type == "general"
    assert result.selected_critics == ["LogicCritic"]


def test_router_node_populates_policy_decision() -> None:
    state = _make_state(task_type="code")

    result = router_node(state)

    assert result.policy_decision is not None
    assert result.policy_decision.action == "select_critics"
    assert result.policy_decision.target_node == NodeName.CRITIC.value
    assert "code" in result.policy_decision.rationale
    assert result.policy_decision.metadata == {
        "task_type": "code",
        "rule": "task_type_code_else_logic",
    }


def test_router_node_policy_decision_reflects_none_task_type() -> None:
    state = _make_state(task_type=None)

    result = router_node(state)

    assert result.policy_decision.metadata["task_type"] is None
    assert result.policy_decision.rationale == "Rule-based selection for task_type=None: ['LogicCritic']."


def test_router_node_ignores_worker_outputs_content() -> None:
    state = _make_state(task_type="research")
    worker_node(state)  # populate worker_outputs; rule should not depend on it

    result = router_node(state)

    assert len(result.worker_outputs) == 1
    assert result.selected_critics == ["LogicCritic"]


def test_router_node_is_deterministic() -> None:
    result_a = router_node(_make_state(task_type="code"))
    result_b = router_node(_make_state(task_type="code"))

    assert result_a.selected_critics == result_b.selected_critics
    assert result_a.policy_decision.model_dump() == result_b.policy_decision.model_dump()


def test_router_node_returns_same_state_instance() -> None:
    state = _make_state(task_type="code")

    result = router_node(state)

    assert result is state


def test_router_node_does_not_modify_other_state_fields() -> None:
    state = _make_state(user_query="Any query", task_type="code")

    result = router_node(state)

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


def test_router_node_overwrites_previous_selected_critics_and_decision() -> None:
    state = _make_state(task_type="code")
    router_node(state)
    assert state.selected_critics == ["CodeCritic"]

    state.task_type = "research"
    router_node(state)

    assert state.selected_critics == ["LogicCritic"]
    assert state.policy_decision.metadata["task_type"] == "research"
