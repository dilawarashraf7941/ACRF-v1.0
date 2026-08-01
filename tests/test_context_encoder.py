"""Unit tests for `ContextEncoder` (app/context/encoder.py)."""

from datetime import datetime, timezone

from app.context import ContextEncoder, ContextVector
from app.context.encoder import UNRECOGNIZED_STATUS_CODE, _build_context_id
from app.experience import ExperienceRecord
from app.state import (
    AgentState,
    ErrorFeature,
    ExecutionStatus,
    PlannerOutput,
    SafetyStatus,
    WorkerOutput,
)

ALL_FEATURE_NAMES = {
    "iteration_count",
    "max_iterations",
    "iteration_ratio",
    "error_feature_count",
    "worker_output_count",
    "critic_score_count",
    "selected_critics_count",
    "retrieved_memories_count",
    "correction_history_count",
    "aggregated_quality_score",
    "has_aggregated_quality_score",
    "safety_status_code",
    "execution_status_code",
    "is_code_task",
    "has_task_type",
    "average_critic_score",
    "max_critic_score",
    "min_critic_score",
    "uncertainty",
    "risk",
    "task_complexity",
    "memory_relevance",
    "requires_self_correction",
    "requires_meta_critic",
    "is_code_output",
    "iteration_pressure",
    "attempt_pressure",
}


def _make_state(**overrides: object) -> AgentState:
    defaults: dict[str, object] = {"session_id": "s1", "task_id": "t1", "user_query": "q"}
    defaults.update(overrides)
    return AgentState(**defaults)  # type: ignore[arg-type]


def _make_experience(**overrides: object) -> ExperienceRecord:
    defaults: dict[str, object] = {
        "experience_id": "exp-1",
        "session_id": "s1",
        "task_id": "t1",
        "timestamp": datetime.now(timezone.utc),
        "iterations": 0,
        "execution_status": "completed",
    }
    defaults.update(overrides)
    return ExperienceRecord(**defaults)  # type: ignore[arg-type]


# --- Basic contract ---


def test_encode_returns_context_vector() -> None:
    context = ContextEncoder().encode(_make_state())

    assert isinstance(context, ContextVector)


def test_encode_produces_every_named_feature() -> None:
    context = ContextEncoder().encode(_make_state())

    assert set(context.features.keys()) == ALL_FEATURE_NAMES
    assert set(context.feature_order) == ALL_FEATURE_NAMES


def test_feature_order_matches_features_keys() -> None:
    context = ContextEncoder().encode(_make_state())

    assert context.feature_order == list(context.features.keys())


def test_neutral_state_produces_all_zero_features_except_max_iterations() -> None:
    """AgentState.max_iterations defaults to 10 (see app/state/state.py); every other feature
    should be 0.0 for a freshly constructed, otherwise-empty state."""
    context = ContextEncoder().encode(_make_state())

    for name, value in context.features.items():
        if name == "max_iterations":
            assert value == 10.0
        else:
            assert value == 0.0, f"{name} was {value}, expected 0.0"


def test_is_not_normalized_by_default() -> None:
    context = ContextEncoder().encode(_make_state())

    assert context.normalized is False
    assert context.normalization_strategy is None


# --- Direct copies / counts ---


def test_iteration_and_max_iterations_are_copied() -> None:
    state = _make_state()
    state.iteration_count = 3
    state.max_iterations = 8

    context = ContextEncoder().encode(state)

    assert context.features["iteration_count"] == 3.0
    assert context.features["max_iterations"] == 8.0


def test_iteration_ratio_is_computed_correctly() -> None:
    state = _make_state()
    state.iteration_count = 2
    state.max_iterations = 8

    context = ContextEncoder().encode(state)

    assert context.features["iteration_ratio"] == 0.25


def test_iteration_ratio_is_zero_when_max_iterations_is_zero() -> None:
    state = _make_state()
    state.max_iterations = 0

    context = ContextEncoder().encode(state)

    assert context.features["iteration_ratio"] == 0.0


def test_counts_reflect_list_and_dict_lengths() -> None:
    state = _make_state()
    state.worker_outputs = [WorkerOutput(worker_id="w"), WorkerOutput(worker_id="w")]
    state.selected_critics = ["LogicCritic", "CodeCritic", "FactCritic"]
    state.critic_scores = {"LogicCritic": 0.5}

    context = ContextEncoder().encode(state)

    assert context.features["worker_output_count"] == 2.0
    assert context.features["selected_critics_count"] == 3.0
    assert context.features["critic_score_count"] == 1.0


# --- Quality score / missingness ---


def test_missing_aggregated_quality_score_is_imputed_with_flag() -> None:
    context = ContextEncoder().encode(_make_state())

    assert context.features["aggregated_quality_score"] == 0.0
    assert context.features["has_aggregated_quality_score"] == 0.0


def test_present_aggregated_quality_score_sets_flag() -> None:
    state = _make_state()
    state.aggregated_quality_score = 0.75

    context = ContextEncoder().encode(state)

    assert context.features["aggregated_quality_score"] == 0.75
    assert context.features["has_aggregated_quality_score"] == 1.0


# --- Status codes ---


def test_safety_status_code_reflects_enum_value() -> None:
    state = _make_state()
    state.safety_status = SafetyStatus.BLOCKED

    context = ContextEncoder().encode(state)

    assert context.features["safety_status_code"] == 3.0


def test_execution_status_code_reflects_enum_value() -> None:
    state = _make_state()
    state.execution_status = ExecutionStatus.FAILED

    context = ContextEncoder().encode(state)

    assert context.features["execution_status_code"] == 4.0


def test_unrecognized_status_string_degrades_gracefully() -> None:
    from app.context.encoder import _status_code

    assert _status_code("some_future_status", {"a": 1.0}) == UNRECOGNIZED_STATUS_CODE


# --- Task type ---


def test_is_code_task_true_for_code_task_type() -> None:
    state = _make_state(task_type="code")

    context = ContextEncoder().encode(state)

    assert context.features["is_code_task"] == 1.0
    assert context.features["has_task_type"] == 1.0


def test_is_code_task_false_for_other_task_type() -> None:
    state = _make_state(task_type="research")

    context = ContextEncoder().encode(state)

    assert context.features["is_code_task"] == 0.0
    assert context.features["has_task_type"] == 1.0


def test_task_type_falls_back_to_planner_output() -> None:
    state = _make_state(task_type=None)
    state.planner_output = PlannerOutput(task_type="code")

    context = ContextEncoder().encode(state)

    assert context.features["is_code_task"] == 1.0
    assert context.features["has_task_type"] == 1.0


def test_task_type_missing_entirely() -> None:
    state = _make_state(task_type=None)

    context = ContextEncoder().encode(state)

    assert context.features["is_code_task"] == 0.0
    assert context.features["has_task_type"] == 0.0


# --- Critic score aggregates ---


def test_critic_score_aggregates_with_multiple_critics() -> None:
    state = _make_state()
    state.critic_scores = {"LogicCritic": 0.2, "CodeCritic": 0.8, "FactCritic": 0.5}

    context = ContextEncoder().encode(state)

    assert context.features["average_critic_score"] == 0.5
    assert context.features["max_critic_score"] == 0.8
    assert context.features["min_critic_score"] == 0.2


def test_critic_score_aggregates_default_to_zero_when_empty() -> None:
    context = ContextEncoder().encode(_make_state())

    assert context.features["average_critic_score"] == 0.0
    assert context.features["max_critic_score"] == 0.0
    assert context.features["min_critic_score"] == 0.0


# --- HeuristicPolicyScorer-parity features (uncertainty, risk, task_complexity,
# memory_relevance, requires_self_correction, requires_meta_critic,
# is_code_output, iteration_pressure, attempt_pressure) ---


def _rich_error_feature() -> ErrorFeature:
    return ErrorFeature(
        error_type="short_output",
        description="test fixture",
        severity="high",
        source_node="error_feature_extractor",
        metadata={
            "confidence": 0.2,
            "risk_level": "high",
            "output_type": "code",
            "error_category": "short_output",
            "profile": {
                "task_complexity": "complex",
                "requires_meta_critic": True,
                "requires_self_correction": True,
                "memory_relevance": 0.9,
                "confidence_score": 0.2,
            },
        },
    )


def test_heuristic_scorer_parity_features_match_extract_features_exactly() -> None:
    """The nine mirrored features must exactly equal
    HeuristicPolicyScorer.extract_features(state) for the same state —
    the property HeuristicPolicy's behavioral identity depends on.
    """
    from app.policy_engine.scorer import HeuristicPolicyScorer

    state = _make_state(task_type="code")
    state.error_features = [_rich_error_feature()]
    state.iteration_count = 3
    state.worker_outputs = [WorkerOutput(worker_id="w"), WorkerOutput(worker_id="w")]

    context = ContextEncoder().encode(state)
    expected = HeuristicPolicyScorer().extract_features(state)

    assert context.features["uncertainty"] == expected.uncertainty
    assert context.features["risk"] == expected.risk
    assert context.features["task_complexity"] == expected.task_complexity
    assert context.features["memory_relevance"] == expected.memory_relevance
    assert bool(context.features["requires_self_correction"]) == expected.requires_self_correction
    assert bool(context.features["requires_meta_critic"]) == expected.requires_meta_critic
    assert bool(context.features["is_code_output"]) == expected.is_code_output
    assert context.features["iteration_pressure"] == expected.iteration_pressure
    assert context.features["attempt_pressure"] == expected.attempt_pressure


def test_heuristic_scorer_parity_features_match_for_neutral_state() -> None:
    from app.policy_engine.scorer import HeuristicPolicyScorer

    state = _make_state()

    context = ContextEncoder().encode(state)
    expected = HeuristicPolicyScorer().extract_features(state)

    assert context.features["uncertainty"] == expected.uncertainty == 0.0
    assert context.features["risk"] == expected.risk == 0.0
    assert context.features["requires_self_correction"] == 0.0
    assert context.features["requires_meta_critic"] == 0.0
    assert context.features["is_code_output"] == 0.0


def test_requires_self_correction_and_meta_critic_flags_from_profile() -> None:
    state = _make_state()
    state.error_features = [_rich_error_feature()]

    context = ContextEncoder().encode(state)

    assert context.features["requires_self_correction"] == 1.0
    assert context.features["requires_meta_critic"] == 1.0


def test_is_code_output_distinct_from_is_code_task() -> None:
    """is_code_output (from error-feature output_type) and is_code_task
    (from task_type) are independent signals that can disagree.
    """
    state = _make_state(task_type="research")  # not a code task...
    state.error_features = [_rich_error_feature()]  # ...but output_type is "code"

    context = ContextEncoder().encode(state)

    assert context.features["is_code_task"] == 0.0
    assert context.features["is_code_output"] == 1.0


def test_attempt_pressure_reflects_worker_output_count() -> None:
    state = _make_state()
    state.worker_outputs = [WorkerOutput(worker_id="w") for _ in range(5)]

    context = ContextEncoder().encode(state)

    # (5 - 1) / 4 == 1.0, matching _ATTEMPT_PRESSURE_CAP
    assert context.features["attempt_pressure"] == 1.0


def test_iteration_pressure_reflects_iteration_count() -> None:
    state = _make_state()
    state.iteration_count = 10  # capped at _ITERATION_PRESSURE_CAP=5

    context = ContextEncoder().encode(state)

    assert context.features["iteration_pressure"] == 1.0


# --- context_id ---


def test_context_id_is_deterministic() -> None:
    state_a = _make_state(session_id="session-x", task_id="task-x")
    state_b = _make_state(session_id="session-x", task_id="task-x")

    context_a = ContextEncoder().encode(state_a)
    context_b = ContextEncoder().encode(state_b)

    assert context_a.context_id == context_b.context_id


def test_context_id_differs_for_different_session() -> None:
    context_a = ContextEncoder().encode(_make_state(session_id="session-a", task_id="task-1"))
    context_b = ContextEncoder().encode(_make_state(session_id="session-b", task_id="task-1"))

    assert context_a.context_id != context_b.context_id


def test_context_id_does_not_collide_with_experience_id_formula() -> None:
    """context_id uses a different (salted) hash namespace than
    ExperienceRecorder's experience_id, even for the same triple.
    """
    import hashlib

    session_id, task_id, iterations = "s1", "t1", 0
    unsalted = hashlib.sha256(f"{session_id}|{task_id}|{iterations}".encode()).hexdigest()

    context_id = _build_context_id(session_id, task_id, iterations)

    assert context_id != unsalted


# --- Optional ExperienceRecord enrichment ---


def test_without_experience_metadata_source_is_agent_state_only() -> None:
    context = ContextEncoder().encode(_make_state())

    assert context.metadata["source"] == "AgentState"
    assert "experience_derived" not in context.metadata
    assert context.source_execution_id is None


def test_with_experience_metadata_records_source_execution_id() -> None:
    experience = _make_experience(experience_id="exp-42")

    context = ContextEncoder().encode(_make_state(), experience=experience)

    assert context.source_execution_id == "exp-42"
    assert context.metadata["source"] == "AgentState+ExperienceRecord"


def test_experience_derived_features_are_not_in_primary_features() -> None:
    """latency/estimated_cost must never leak into the pre-decision features dict."""
    experience = _make_experience(latency=2.5, estimated_cost=0.1)

    context = ContextEncoder().encode(_make_state(), experience=experience)

    assert "latency" not in context.features
    assert "estimated_cost" not in context.features
    assert context.metadata["experience_derived"]["latency"] == 2.5
    assert context.metadata["experience_derived"]["estimated_cost"] == 0.1
    assert context.metadata["experience_derived"]["has_latency"] == 1.0


def test_experience_derived_features_degrade_gracefully_when_missing() -> None:
    experience = _make_experience(latency=None, estimated_cost=None)

    context = ContextEncoder().encode(_make_state(), experience=experience)

    assert context.metadata["experience_derived"]["latency"] == 0.0
    assert context.metadata["experience_derived"]["has_latency"] == 0.0
    assert context.metadata["experience_derived"]["estimated_cost"] == 0.0
    assert context.metadata["experience_derived"]["has_estimated_cost"] == 0.0


def test_timestamp_uses_experience_timestamp_when_provided() -> None:
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    experience = _make_experience(timestamp=ts)

    context = ContextEncoder().encode(_make_state(), experience=experience)

    assert context.timestamp == ts


def test_timestamp_uses_execution_metadata_when_no_experience() -> None:
    state = _make_state()

    context = ContextEncoder().encode(state)

    assert context.timestamp == state.execution_metadata.updated_at


# --- Read-only behavior ---


def test_encode_does_not_mutate_state() -> None:
    state = _make_state()
    state.critic_scores = {"LogicCritic": 0.5}
    snapshot = state.model_dump(mode="json")

    ContextEncoder().encode(state)

    assert state.model_dump(mode="json") == snapshot


def test_encode_does_not_mutate_experience() -> None:
    experience = _make_experience()
    snapshot = experience.model_dump(mode="json")

    ContextEncoder().encode(_make_state(), experience=experience)

    assert experience.model_dump(mode="json") == snapshot


# --- Determinism ---


def test_encode_is_deterministic() -> None:
    state = _make_state()
    state.aggregated_quality_score = 0.6
    state.critic_scores = {"LogicCritic": 0.4}

    context_a = ContextEncoder().encode(state)
    context_b = ContextEncoder().encode(state)

    assert context_a == context_b
