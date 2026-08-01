"""Unit tests for the deterministic placeholder `critic_node` (Algorithm 1,
steps 7-8: execute the selected critic set and aggregate feedback).
"""

from app.graph.nodes import critic_node
from app.state import AgentState, ExecutionStatus, SafetyStatus, WorkerOutput


def _make_state(selected_critics: list[str], output: str = "some output") -> AgentState:
    state = AgentState(session_id="session-1", task_id="task-1", user_query="q")
    state.worker_outputs = [WorkerOutput(worker_id="worker-001", output=output)]
    state.selected_critics = selected_critics
    return state


def test_executes_each_selected_critic() -> None:
    state = _make_state(["LogicCritic", "CodeCritic"])

    result = critic_node(state)

    assert set(result.critic_scores.keys()) == {"LogicCritic", "CodeCritic"}
    assert len(result.critic_feedback) == 2


def test_placeholder_critic_scores_are_neutral() -> None:
    state = _make_state(["LogicCritic"])

    result = critic_node(state)

    assert result.critic_scores["LogicCritic"] == 0.0


def test_critic_feedback_entries_have_expected_shape() -> None:
    state = _make_state(["FactCritic"])

    result = critic_node(state)

    entry = result.critic_feedback[0]
    assert entry.critic_name == "FactCritic"
    assert "placeholder" in entry.feedback.lower()
    assert entry.metadata["critic_class"] == "FactCritic"


def test_unknown_critic_name_is_skipped() -> None:
    state = _make_state(["LogicCritic", "NotARealCritic"])

    result = critic_node(state)

    assert set(result.critic_scores.keys()) == {"LogicCritic"}
    assert len(result.critic_feedback) == 1


def test_empty_selected_critics_produces_no_results_but_valid_aggregation() -> None:
    state = _make_state([])

    result = critic_node(state)

    assert result.critic_feedback == []
    assert result.critic_scores == {}
    assert result.aggregated_quality_score == 0.0
    assert result.memory_context["critic_aggregation"]["contributing_critics"] == []


def test_sets_aggregated_quality_score_from_placeholder_aggregation() -> None:
    state = _make_state(["LogicCritic", "CodeCritic", "FactCritic", "MetaCritic"])

    result = critic_node(state)

    assert result.aggregated_quality_score == 0.0
    aggregation = result.memory_context["critic_aggregation"]
    assert aggregation["strategy_name"] == "MajorityVoteStrategy"
    assert aggregation["contributing_critics"] == ["LogicCritic", "CodeCritic", "FactCritic", "MetaCritic"]


def test_critic_scores_do_not_depend_on_worker_output_content() -> None:
    """No evaluation logic: differing worker output text must not change
    the (fixed, neutral) critic scores.
    """
    result_a = critic_node(_make_state(["LogicCritic"], output="short"))
    result_b = critic_node(
        _make_state(["LogicCritic"], output="a very very long and detailed piece of text output")
    )

    assert result_a.critic_scores == result_b.critic_scores == {"LogicCritic": 0.0}


def test_appends_to_existing_critic_feedback_without_discarding() -> None:
    state = _make_state(["LogicCritic"])

    critic_node(state)
    critic_node(state)

    assert len(state.critic_feedback) == 2


def test_merges_into_existing_critic_scores() -> None:
    state = _make_state(["LogicCritic"])
    state.critic_scores = {"SomeOtherCritic": 0.75}

    result = critic_node(state)

    assert result.critic_scores == {"SomeOtherCritic": 0.75, "LogicCritic": 0.0}


def test_preserves_existing_memory_context_keys() -> None:
    state = _make_state(["LogicCritic"])
    state.memory_context = {"existing": "value"}

    result = critic_node(state)

    assert result.memory_context["existing"] == "value"
    assert "critic_aggregation" in result.memory_context


def test_returns_same_state_instance() -> None:
    state = _make_state(["LogicCritic"])

    result = critic_node(state)

    assert result is state


def test_does_not_modify_unrelated_state_fields() -> None:
    state = _make_state(["LogicCritic"])

    result = critic_node(state)

    assert result.correction_history == []
    assert result.safety_status == SafetyStatus.UNKNOWN
    assert result.final_response is None
    assert result.execution_status == ExecutionStatus.PENDING


def test_is_deterministic() -> None:
    critics = ["LogicCritic", "CodeCritic"]

    result_a = critic_node(_make_state(critics))
    result_b = critic_node(_make_state(critics))

    assert result_a.critic_scores == result_b.critic_scores
    assert result_a.aggregated_quality_score == result_b.aggregated_quality_score
    assert [f.model_dump() for f in result_a.critic_feedback] == [
        f.model_dump() for f in result_b.critic_feedback
    ]
