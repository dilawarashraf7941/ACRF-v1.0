"""Unit tests for `ExperienceRecord` (app/experience/models.py)."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.experience import ExperienceRecord


def _make_record(**overrides: object) -> ExperienceRecord:
    defaults: dict[str, object] = {
        "experience_id": "abc123",
        "session_id": "s1",
        "task_id": "t1",
        "timestamp": datetime.now(timezone.utc),
        "iterations": 0,
        "execution_status": "completed",
    }
    defaults.update(overrides)
    return ExperienceRecord(**defaults)  # type: ignore[arg-type]


def test_requires_core_fields() -> None:
    with pytest.raises(ValidationError):
        ExperienceRecord()  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "field",
    ["experience_id", "session_id", "task_id", "timestamp", "iterations", "execution_status"],
)
def test_required_fields_are_enforced(field: str) -> None:
    kwargs = {
        "experience_id": "abc123",
        "session_id": "s1",
        "task_id": "t1",
        "timestamp": datetime.now(timezone.utc),
        "iterations": 0,
        "execution_status": "completed",
    }
    del kwargs[field]

    with pytest.raises(ValidationError):
        ExperienceRecord(**kwargs)  # type: ignore[arg-type]


def test_applies_defaults_for_optional_fields() -> None:
    record = _make_record()

    assert record.state_features == {}
    assert record.selected_critics == []
    assert record.critic_scores == {}
    assert record.aggregated_quality_score is None
    assert record.correction_decision is None
    assert record.final_response is None
    assert record.latency is None
    assert record.estimated_cost is None
    assert record.memory_usage == {}
    assert record.metadata == {}


def test_accepts_all_fields_explicitly() -> None:
    now = datetime.now(timezone.utc)

    record = _make_record(
        timestamp=now,
        state_features={"task_type": "code"},
        selected_critics=["CodeCritic"],
        critic_scores={"CodeCritic": 0.5},
        aggregated_quality_score=0.5,
        correction_decision={"should_correct": True},
        iterations=3,
        final_response="done",
        execution_status="completed",
        latency=1.23,
        estimated_cost=0.0,
        memory_usage={"retrieved_memories_count": 0},
        metadata={"source_node": "evaluation_node"},
    )

    assert record.timestamp == now
    assert record.state_features == {"task_type": "code"}
    assert record.selected_critics == ["CodeCritic"]
    assert record.critic_scores == {"CodeCritic": 0.5}
    assert record.aggregated_quality_score == 0.5
    assert record.correction_decision == {"should_correct": True}
    assert record.iterations == 3
    assert record.final_response == "done"
    assert record.latency == 1.23
    assert record.estimated_cost == 0.0
    assert record.memory_usage == {"retrieved_memories_count": 0}
    assert record.metadata == {"source_node": "evaluation_node"}


def test_iterations_must_be_non_negative() -> None:
    with pytest.raises(ValidationError):
        _make_record(iterations=-1)


def test_is_frozen() -> None:
    record = _make_record()

    with pytest.raises(ValidationError):
        record.session_id = "changed"  # type: ignore[misc]


def test_allows_extra_fields() -> None:
    record = _make_record(custom_field="value")

    assert record.custom_field == "value"  # type: ignore[attr-defined]


def test_round_trips_via_model_dump() -> None:
    record = _make_record(critic_scores={"LogicCritic": 0.7})

    dumped = record.model_dump(mode="json")

    assert dumped["experience_id"] == "abc123"
    assert dumped["critic_scores"] == {"LogicCritic": 0.7}
    reconstructed = ExperienceRecord(**dumped)
    assert reconstructed == record
