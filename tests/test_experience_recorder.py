"""Unit tests for `ExperienceRecorder` (app/experience/recorder.py)."""

from app.experience import ExperienceRecord, ExperienceRecorder, InMemoryExperienceRepository
from app.state import AgentState, ErrorFeature, ExecutionStatus, PlannerOutput, WorkerOutput


def _make_state(session_id: str = "s1", task_id: str = "t1", **overrides: object) -> AgentState:
    defaults: dict[str, object] = {"session_id": session_id, "task_id": task_id, "user_query": "q"}
    defaults.update(overrides)
    return AgentState(**defaults)  # type: ignore[arg-type]


# --- Correctness: field-by-field transcription ---


def test_record_returns_experience_record() -> None:
    record = ExperienceRecorder().record(_make_state())

    assert isinstance(record, ExperienceRecord)


def test_record_copies_identity_fields() -> None:
    state = _make_state(session_id="session-42", task_id="task-7")

    record = ExperienceRecorder().record(state)

    assert record.session_id == "session-42"
    assert record.task_id == "task-7"


def test_record_copies_critique_fields() -> None:
    state = _make_state()
    state.selected_critics = ["LogicCritic", "CodeCritic"]
    state.critic_scores = {"LogicCritic": 0.4, "CodeCritic": 0.6}
    state.aggregated_quality_score = 0.5

    record = ExperienceRecorder().record(state)

    assert record.selected_critics == ["LogicCritic", "CodeCritic"]
    assert record.critic_scores == {"LogicCritic": 0.4, "CodeCritic": 0.6}
    assert record.aggregated_quality_score == 0.5


def test_record_copies_iterations_final_response_and_status() -> None:
    state = _make_state()
    state.iteration_count = 2
    state.final_response = "the answer"
    state.execution_status = ExecutionStatus.COMPLETED

    record = ExperienceRecorder().record(state)

    assert record.iterations == 2
    assert record.final_response == "the answer"
    assert record.execution_status == "completed"


def test_record_extracts_correction_decision_from_memory_context() -> None:
    state = _make_state()
    state.memory_context = {
        "correction_policy": {"decision": {"should_correct": True, "confidence": 0.75}}
    }

    record = ExperienceRecorder().record(state)

    assert record.correction_decision == {"should_correct": True, "confidence": 0.75}


def test_record_correction_decision_is_none_when_absent() -> None:
    record = ExperienceRecorder().record(_make_state())

    assert record.correction_decision is None


def test_record_state_features_includes_task_type_and_error_features() -> None:
    state = _make_state(task_type="code")
    state.error_features = [
        ErrorFeature(error_type="x", description="d", metadata={"risk_level": "high"})
    ]
    state.planner_output = PlannerOutput(decomposition=["a", "b"])

    record = ExperienceRecorder().record(state)

    assert record.state_features["task_type"] == "code"
    assert record.state_features["error_feature_count"] == 1
    assert record.state_features["error_features"][0]["metadata"]["risk_level"] == "high"
    assert record.state_features["planner_output"]["decomposition"] == ["a", "b"]


def test_record_state_features_planner_output_none_when_absent() -> None:
    record = ExperienceRecorder().record(_make_state())

    assert record.state_features["planner_output"] is None


def test_record_memory_usage_reflects_state() -> None:
    state = _make_state()
    state.memory_context = {"foo": "bar", "baz": 1}

    record = ExperienceRecorder().record(state)

    assert record.memory_usage["retrieved_memories_count"] == 0
    assert record.memory_usage["memory_context_keys"] == ["baz", "foo"]


def test_record_estimated_cost_sums_worker_token_usage() -> None:
    state = _make_state()
    state.worker_outputs = [
        WorkerOutput(worker_id="w", token_usage=10),
        WorkerOutput(worker_id="w", token_usage=5),
    ]

    record = ExperienceRecorder().record(state)

    assert record.estimated_cost == 15.0


def test_record_estimated_cost_defaults_to_zero_without_worker_outputs() -> None:
    record = ExperienceRecorder().record(_make_state())

    assert record.estimated_cost == 0.0


def test_record_estimated_cost_ignores_missing_token_usage() -> None:
    state = _make_state()
    state.worker_outputs = [WorkerOutput(worker_id="w")]  # no token_usage set

    record = ExperienceRecorder().record(state)

    assert record.estimated_cost == 0.0


def test_record_latency_reflects_execution_metadata_timestamps() -> None:
    from datetime import datetime, timedelta, timezone

    from app.state import ExecutionMetadata

    state = _make_state()
    created = datetime(2026, 1, 1, tzinfo=timezone.utc)
    updated = created + timedelta(seconds=2.5)
    state.execution_metadata = ExecutionMetadata(created_at=created, updated_at=updated)

    record = ExperienceRecorder().record(state)

    assert record.latency == 2.5


def test_record_timestamp_matches_execution_metadata_updated_at() -> None:
    state = _make_state()

    record = ExperienceRecorder().record(state)

    assert record.timestamp == state.execution_metadata.updated_at


# --- Deterministic IDs ---


def test_experience_id_is_deterministic_for_identical_inputs() -> None:
    state_a = _make_state("same-session", "same-task")
    state_b = _make_state("same-session", "same-task")

    record_a = ExperienceRecorder().record(state_a)
    record_b = ExperienceRecorder().record(state_b)

    assert record_a.experience_id == record_b.experience_id


def test_experience_id_differs_for_different_session_id() -> None:
    record_a = ExperienceRecorder().record(_make_state("session-a", "task-1"))
    record_b = ExperienceRecorder().record(_make_state("session-b", "task-1"))

    assert record_a.experience_id != record_b.experience_id


def test_experience_id_differs_for_different_task_id() -> None:
    record_a = ExperienceRecorder().record(_make_state("session-1", "task-a"))
    record_b = ExperienceRecorder().record(_make_state("session-1", "task-b"))

    assert record_a.experience_id != record_b.experience_id


def test_experience_id_differs_for_different_iteration_count() -> None:
    state_a = _make_state("session-1", "task-1")
    state_a.iteration_count = 0
    state_b = _make_state("session-1", "task-1")
    state_b.iteration_count = 1

    record_a = ExperienceRecorder().record(state_a)
    record_b = ExperienceRecorder().record(state_b)

    assert record_a.experience_id != record_b.experience_id


def test_no_duplicate_ids_across_many_distinct_states() -> None:
    ids = set()
    for i in range(50):
        state = _make_state(f"session-{i}", f"task-{i}")
        state.iteration_count = i
        ids.add(ExperienceRecorder().record(state).experience_id)

    assert len(ids) == 50


# --- Read-only behavior ---


def test_record_does_not_mutate_state() -> None:
    state = _make_state()
    state.selected_critics = ["LogicCritic"]
    state.critic_scores = {"LogicCritic": 0.5}
    snapshot = state.model_dump(mode="json")

    ExperienceRecorder().record(state)

    assert state.model_dump(mode="json") == snapshot


def test_record_returns_independent_copies_of_mutable_state_fields() -> None:
    state = _make_state()
    state.selected_critics = ["LogicCritic"]

    record = ExperienceRecorder().record(state)
    record.selected_critics.append("CodeCritic")

    assert state.selected_critics == ["LogicCritic"]


# --- Dependency injection into a repository ---


def test_record_without_repository_does_not_store_anything() -> None:
    ExperienceRecorder(repository=None).record(_make_state())
    # Nothing to assert against directly; this just confirms no exception
    # is raised when no repository is injected.


def test_record_with_injected_repository_stores_the_record() -> None:
    repository = InMemoryExperienceRepository()
    recorder = ExperienceRecorder(repository=repository)

    record = recorder.record(_make_state())

    assert repository.count() == 1
    assert repository.get(record.experience_id) == record


def test_record_with_injected_repository_returns_the_built_record() -> None:
    repository = InMemoryExperienceRepository()
    recorder = ExperienceRecorder(repository=repository)

    record = recorder.record(_make_state())

    assert record == repository.get(record.experience_id)
