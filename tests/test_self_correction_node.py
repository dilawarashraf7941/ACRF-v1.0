"""Unit tests for `self_correction_node` (Algorithm 1, step 9), which now
delegates the "should we correct?" question to `CorrectionDecisionEngine`
(see `app/correction_policy`) instead of always correcting.
"""

from app.graph.nodes import NodeName, self_correction_node
from app.state import AgentState, ErrorFeature, ExecutionStatus, SafetyStatus, WorkerOutput


def _make_state(user_query: str = "q") -> AgentState:
    state = AgentState(session_id="session-1", task_id="task-1", user_query=user_query)
    state.worker_outputs = [WorkerOutput(worker_id="worker-001", output="original response")]
    return state


def _low_quality_state() -> AgentState:
    state = _make_state()
    state.aggregated_quality_score = 0.2
    return state


# --- should_correct == False: no correction applied ---


def test_neutral_state_does_not_correct() -> None:
    """No signal at all (no quality score, no critic scores, no error
    features) must default to no correction.
    """
    state = _make_state()

    result = self_correction_node(state)

    assert len(result.worker_outputs) == 1
    assert result.correction_history == []
    assert result.iteration_count == 0


def test_high_quality_skips_correction() -> None:
    state = _make_state()
    state.aggregated_quality_score = 0.95
    state.critic_scores = {"LogicCritic": 0.9, "CodeCritic": 0.85}

    result = self_correction_node(state)

    assert len(result.worker_outputs) == 1
    assert result.correction_history == []
    assert result.iteration_count == 0
    assert result.memory_context["correction_policy"]["strategy"] == "rule_based_finish"


def test_max_iterations_prevents_correction_even_with_low_quality() -> None:
    state = _low_quality_state()
    state.iteration_count = 10
    state.max_iterations = 10

    result = self_correction_node(state)

    assert len(result.worker_outputs) == 1
    assert result.correction_history == []
    assert result.iteration_count == 10
    assert result.memory_context["correction_policy"]["strategy"] == "hard_stop_max_iterations"


def test_no_correction_returns_state_otherwise_unchanged() -> None:
    state = _make_state()

    result = self_correction_node(state)

    assert result.critic_feedback == []
    assert result.critic_scores == {}
    assert result.safety_status == SafetyStatus.UNKNOWN
    assert result.final_response is None
    assert result.execution_status == ExecutionStatus.PENDING


# --- should_correct == True: correction applied ---


def test_low_quality_triggers_correction() -> None:
    state = _low_quality_state()

    result = self_correction_node(state)

    assert len(result.worker_outputs) == 2
    assert len(result.correction_history) == 1
    assert result.iteration_count == 1
    assert result.memory_context["correction_policy"]["strategy"] == "rule_based_correction"


def test_self_correction_error_feature_triggers_correction() -> None:
    state = _make_state()
    state.error_features = [
        ErrorFeature(
            error_type="x",
            description="d",
            metadata={"profile": {"requires_self_correction": True}},
        )
    ]

    result = self_correction_node(state)

    assert len(result.worker_outputs) == 2
    triggered_rules = result.memory_context["correction_policy"]["triggered_rules"]
    assert "requires_self_correction" in triggered_rules


def test_meta_critic_escalation_triggers_correction() -> None:
    state = _make_state()
    state.critic_scores = {"MetaCritic": 0.9}

    result = self_correction_node(state)

    assert len(result.worker_outputs) == 2
    assert "meta_critic_escalation" in result.memory_context["correction_policy"]["triggered_rules"]


def test_appends_correction_record() -> None:
    state = _low_quality_state()

    result = self_correction_node(state)

    assert len(result.correction_history) == 1
    record = result.correction_history[0]
    assert record.iteration == 0
    assert "placeholder" in record.description.lower()
    assert record.applied_by == NodeName.SELF_CORRECTION.value


def test_increments_iteration_count() -> None:
    state = _low_quality_state()
    assert state.iteration_count == 0

    result = self_correction_node(state)

    assert result.iteration_count == 1


def test_increments_iteration_count_from_nonzero_start() -> None:
    state = _low_quality_state()
    state.iteration_count = 3
    state.max_iterations = 10

    result = self_correction_node(state)

    assert result.iteration_count == 4
    assert result.correction_history[0].iteration == 3


def test_appends_new_worker_output_with_corrected_status() -> None:
    state = _low_quality_state()

    result = self_correction_node(state)

    assert len(result.worker_outputs) == 2
    corrected = result.worker_outputs[-1]
    assert corrected.status == "corrected"
    assert corrected.output == "Placeholder corrected response."
    assert corrected.metadata["corrected_at_iteration"] == 1


def test_corrected_worker_output_input_uses_resolved_worker_input() -> None:
    from app.graph.nodes import planner_node

    state = _low_quality_state()
    planner_node(state)  # sets planner_output.original_query = "q"

    result = self_correction_node(state)

    assert result.worker_outputs[-1].input == "q"


def test_repeated_calls_accumulate_history_and_outputs() -> None:
    state = _low_quality_state()

    self_correction_node(state)
    self_correction_node(state)

    assert state.iteration_count == 2
    assert len(state.correction_history) == 2
    assert len(state.worker_outputs) == 3
    assert [r.iteration for r in state.correction_history] == [0, 1]


def test_correction_does_not_depend_on_worker_output_content() -> None:
    """The correction itself performs no reasoning: differing existing
    output text must not change the fixed placeholder correction, only
    whether the decision policy triggers it (governed separately by
    aggregated_quality_score/critic_scores/etc.).
    """
    state_a = _low_quality_state()
    state_a.worker_outputs = [WorkerOutput(worker_id="worker-001", output="x")]
    state_b = _low_quality_state()
    state_b.worker_outputs = [
        WorkerOutput(worker_id="worker-001", output="a totally different value")
    ]

    result_a = self_correction_node(state_a)
    result_b = self_correction_node(state_b)

    assert result_a.worker_outputs[-1].output == result_b.worker_outputs[-1].output
    assert result_a.correction_history[0].description == result_b.correction_history[0].description


# --- Diagnostics: always recorded, regardless of the decision ---


def test_diagnostics_recorded_when_not_correcting() -> None:
    state = _make_state()

    result = self_correction_node(state)

    diagnostics = result.memory_context["correction_policy"]
    assert diagnostics["decision"]["should_correct"] is False
    assert diagnostics["strategy"] == "default_no_signal"
    assert diagnostics["confidence"] == 0.0
    assert diagnostics["triggered_rules"] == []


def test_diagnostics_recorded_when_correcting() -> None:
    state = _low_quality_state()

    result = self_correction_node(state)

    diagnostics = result.memory_context["correction_policy"]
    assert diagnostics["decision"]["should_correct"] is True
    assert diagnostics["strategy"] == "rule_based_correction"
    assert diagnostics["confidence"] > 0.0
    assert "low_aggregated_quality" in diagnostics["triggered_rules"]


def test_diagnostics_include_full_rule_results() -> None:
    state = _low_quality_state()

    result = self_correction_node(state)

    decision_metadata = result.memory_context["correction_policy"]["decision"]["metadata"]
    rule_results = decision_metadata["rule_results"]
    assert {entry["rule_name"] for entry in rule_results} == {
        "max_iterations_reached",
        "low_aggregated_quality",
        "meta_critic_escalation",
        "requires_self_correction",
        "low_memory_relevance",
        "all_critics_high_quality",
    }


def test_preserves_existing_memory_context_keys() -> None:
    state = _low_quality_state()
    state.memory_context = {"existing": "value"}

    result = self_correction_node(state)

    assert result.memory_context["existing"] == "value"
    assert "correction_policy" in result.memory_context


# --- Determinism ---


def test_is_deterministic() -> None:
    result_a = self_correction_node(_low_quality_state())
    result_b = self_correction_node(_low_quality_state())

    assert result_a.iteration_count == result_b.iteration_count
    assert len(result_a.worker_outputs) == len(result_b.worker_outputs)
    diagnostics_a = result_a.memory_context["correction_policy"]["decision"]
    diagnostics_b = result_b.memory_context["correction_policy"]["decision"]
    assert diagnostics_a == diagnostics_b


def test_is_deterministic_for_no_correction_case() -> None:
    result_a = self_correction_node(_make_state())
    result_b = self_correction_node(_make_state())

    diagnostics_a = result_a.memory_context["correction_policy"]
    diagnostics_b = result_b.memory_context["correction_policy"]
    assert diagnostics_a == diagnostics_b


# --- General node behavior ---


def test_returns_same_state_instance() -> None:
    state = _make_state()

    result = self_correction_node(state)

    assert result is state
