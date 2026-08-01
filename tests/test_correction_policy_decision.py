"""Unit tests for `CorrectionDecisionEngine` (app/correction_policy/decision.py)."""

import pytest
from pydantic import ValidationError

from app.correction_policy import CorrectionDecision, CorrectionDecisionEngine
from app.state import AgentState, ErrorFeature, PlannerOutput, WorkerOutput


def _make_state(**overrides: object) -> AgentState:
    defaults: dict[str, object] = {"session_id": "s", "task_id": "t", "user_query": "q"}
    defaults.update(overrides)
    return AgentState(**defaults)  # type: ignore[arg-type]


def _error_feature_with_flag(requires_self_correction: bool) -> ErrorFeature:
    return ErrorFeature(
        error_type="x",
        description="test fixture",
        metadata={"profile": {"requires_self_correction": requires_self_correction}},
    )


# --- Individual rule triggers, exercised through the engine ---


def test_low_quality_triggers_correction() -> None:
    state = _make_state()
    state.aggregated_quality_score = 0.2

    decision = CorrectionDecisionEngine().decide(state)

    assert decision.should_correct is True
    assert decision.decision_strategy == "rule_based_correction"
    assert "low_aggregated_quality" in decision.triggered_rules


def test_high_quality_skips_correction_via_finish_rule() -> None:
    state = _make_state()
    state.aggregated_quality_score = 0.95
    state.critic_scores = {"LogicCritic": 0.9, "CodeCritic": 0.85}

    decision = CorrectionDecisionEngine().decide(state)

    assert decision.should_correct is False
    assert decision.decision_strategy == "rule_based_finish"
    assert "all_critics_high_quality" in decision.triggered_rules


def test_max_iterations_prevents_correction() -> None:
    state = _make_state()
    state.iteration_count = 5
    state.max_iterations = 5

    decision = CorrectionDecisionEngine().decide(state)

    assert decision.should_correct is False
    assert decision.decision_strategy == "hard_stop_max_iterations"
    assert decision.confidence == 1.0


def test_max_iterations_overrides_low_quality() -> None:
    """Even a strong "correct" signal must not override the hard stop."""
    state = _make_state()
    state.aggregated_quality_score = 0.0
    state.critic_scores = {"MetaCritic": 1.0}
    state.iteration_count = 3
    state.max_iterations = 3

    decision = CorrectionDecisionEngine().decide(state)

    assert decision.should_correct is False
    assert decision.decision_strategy == "hard_stop_max_iterations"
    # Diagnostics still surface every rule that matched, even though it lost priority.
    assert "low_aggregated_quality" in decision.triggered_rules
    assert "meta_critic_escalation" in decision.triggered_rules


def test_self_correction_feature_triggers_correction() -> None:
    state = _make_state()
    state.error_features = [_error_feature_with_flag(True)]

    decision = CorrectionDecisionEngine().decide(state)

    assert decision.should_correct is True
    assert "requires_self_correction" in decision.triggered_rules


def test_self_correction_feature_false_does_not_trigger() -> None:
    state = _make_state()
    state.error_features = [_error_feature_with_flag(False)]

    decision = CorrectionDecisionEngine().decide(state)

    assert decision.should_correct is False
    assert "requires_self_correction" not in decision.triggered_rules


def test_meta_critic_trigger() -> None:
    state = _make_state()
    state.critic_scores = {"MetaCritic": 0.85}

    decision = CorrectionDecisionEngine().decide(state)

    assert decision.should_correct is True
    assert "meta_critic_escalation" in decision.triggered_rules


def test_low_memory_relevance_triggers_correction() -> None:
    state = _make_state()
    state.memory_context = {"memory_relevance": 0.05}

    decision = CorrectionDecisionEngine().decide(state)

    assert decision.should_correct is True
    assert "low_memory_relevance" in decision.triggered_rules


def test_no_signal_defaults_to_no_correction() -> None:
    state = _make_state()

    decision = CorrectionDecisionEngine().decide(state)

    assert decision.should_correct is False
    assert decision.decision_strategy == "default_no_signal"
    assert decision.confidence == 0.0
    assert decision.triggered_rules == []


# --- Confidence scaling ---


def test_confidence_scales_with_number_of_triggered_correction_rules() -> None:
    single_signal = _make_state()
    single_signal.aggregated_quality_score = 0.1

    multi_signal = _make_state()
    multi_signal.aggregated_quality_score = 0.1
    multi_signal.critic_scores = {"MetaCritic": 0.9}

    single_decision = CorrectionDecisionEngine().decide(single_signal)
    multi_decision = CorrectionDecisionEngine().decide(multi_signal)

    assert multi_decision.confidence > single_decision.confidence


def test_confidence_is_bounded_at_one() -> None:
    state = _make_state()
    state.aggregated_quality_score = 0.0
    state.critic_scores = {"MetaCritic": 1.0}
    state.memory_context = {"memory_relevance": 0.0}
    state.error_features = [_error_feature_with_flag(True)]

    decision = CorrectionDecisionEngine().decide(state)

    assert decision.confidence <= 1.0


# --- Diagnostics ---


def test_decision_metadata_contains_all_rule_results() -> None:
    state = _make_state()

    decision = CorrectionDecisionEngine().decide(state)

    rule_names = {entry["rule_name"] for entry in decision.metadata["rule_results"]}
    assert rule_names == {
        "max_iterations_reached",
        "low_aggregated_quality",
        "meta_critic_escalation",
        "requires_self_correction",
        "low_memory_relevance",
        "all_critics_high_quality",
    }


def test_correction_decision_metadata_lists_correcting_rules() -> None:
    state = _make_state()
    state.aggregated_quality_score = 0.1
    state.critic_scores = {"MetaCritic": 0.9}

    decision = CorrectionDecisionEngine().decide(state)

    assert set(decision.metadata["correcting_rules"]) == {
        "low_aggregated_quality",
        "meta_critic_escalation",
    }


# --- Read-only contract: only the six declared fields matter ---


def test_engine_does_not_read_task_type() -> None:
    with_task_type = _make_state(task_type="code")
    without_task_type = _make_state(task_type=None)

    decision_a = CorrectionDecisionEngine().decide(with_task_type)
    decision_b = CorrectionDecisionEngine().decide(without_task_type)

    dump_a = decision_a.model_dump(exclude={"metadata"})
    dump_b = decision_b.model_dump(exclude={"metadata"})
    assert dump_a == dump_b


def test_engine_does_not_read_planner_output_or_worker_outputs() -> None:
    """These fields are not in the engine's declared input set; changing
    them alone must not change the decision.
    """
    plain = _make_state()
    with_extras = _make_state()
    with_extras.planner_output = PlannerOutput(decomposition=["a", "b", "c"])
    with_extras.worker_outputs = [WorkerOutput(worker_id="w") for _ in range(5)]

    decision_plain = CorrectionDecisionEngine().decide(plain)
    decision_extras = CorrectionDecisionEngine().decide(with_extras)

    assert decision_plain.should_correct == decision_extras.should_correct
    assert decision_plain.decision_strategy == decision_extras.decision_strategy


# --- Determinism ---


def test_is_deterministic() -> None:
    state_a = _make_state()
    state_a.aggregated_quality_score = 0.3
    state_b = _make_state()
    state_b.aggregated_quality_score = 0.3

    decision_a = CorrectionDecisionEngine().decide(state_a)
    decision_b = CorrectionDecisionEngine().decide(state_b)

    assert decision_a.model_dump() == decision_b.model_dump()


# --- CorrectionDecision model shape ---


def test_confidence_is_within_bounds_pydantic_validation() -> None:
    with pytest.raises(ValidationError):
        CorrectionDecision(
            should_correct=True,
            reason="x",
            confidence=1.5,
            decision_strategy="test",
        )


def test_correction_decision_allows_extra_fields() -> None:
    decision = CorrectionDecision(
        should_correct=False,
        reason="x",
        confidence=0.5,
        decision_strategy="test",
        custom_field="value",
    )

    assert decision.custom_field == "value"  # type: ignore[attr-defined]
