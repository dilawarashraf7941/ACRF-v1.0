"""Unit tests for `InMemoryExperienceRepository` (app/experience/repository.py)."""

from datetime import datetime, timezone

import pytest

from app.experience import ExperienceRecord, ExperienceRepository, InMemoryExperienceRepository


def _make_record(experience_id: str = "id-1", **overrides: object) -> ExperienceRecord:
    defaults: dict[str, object] = {
        "experience_id": experience_id,
        "session_id": "s1",
        "task_id": "t1",
        "timestamp": datetime.now(timezone.utc),
        "iterations": 0,
        "execution_status": "completed",
    }
    defaults.update(overrides)
    return ExperienceRecord(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def repository() -> InMemoryExperienceRepository:
    return InMemoryExperienceRepository()


def test_new_repository_is_empty(repository: InMemoryExperienceRepository) -> None:
    assert repository.count() == 0
    assert repository.list() == []


def test_add_then_count(repository: InMemoryExperienceRepository) -> None:
    repository.add(_make_record())

    assert repository.count() == 1


def test_add_multiple_distinct_records(repository: InMemoryExperienceRepository) -> None:
    repository.add(_make_record("id-1"))
    repository.add(_make_record("id-2"))
    repository.add(_make_record("id-3"))

    assert repository.count() == 3


def test_add_rejects_duplicate_experience_id(repository: InMemoryExperienceRepository) -> None:
    repository.add(_make_record("id-1"))

    with pytest.raises(ValueError):
        repository.add(_make_record("id-1"))

    assert repository.count() == 1


def test_get_returns_stored_record(repository: InMemoryExperienceRepository) -> None:
    record = _make_record("id-1")
    repository.add(record)

    assert repository.get("id-1") == record


def test_get_unknown_id_returns_none(repository: InMemoryExperienceRepository) -> None:
    assert repository.get("does-not-exist") is None


def test_list_returns_all_records(repository: InMemoryExperienceRepository) -> None:
    record_a = _make_record("id-1")
    record_b = _make_record("id-2")
    repository.add(record_a)
    repository.add(record_b)

    listed = repository.list()

    assert len(listed) == 2
    assert record_a in listed
    assert record_b in listed


def test_list_preserves_insertion_order(repository: InMemoryExperienceRepository) -> None:
    repository.add(_make_record("id-3"))
    repository.add(_make_record("id-1"))
    repository.add(_make_record("id-2"))

    assert [r.experience_id for r in repository.list()] == ["id-3", "id-1", "id-2"]


def test_list_returns_a_copy_not_internal_state(repository: InMemoryExperienceRepository) -> None:
    repository.add(_make_record("id-1"))

    listed = repository.list()
    listed.append(_make_record("id-2"))

    assert repository.count() == 1


def test_clear_empties_the_repository(repository: InMemoryExperienceRepository) -> None:
    repository.add(_make_record("id-1"))
    repository.add(_make_record("id-2"))

    repository.clear()

    assert repository.count() == 0
    assert repository.list() == []
    assert repository.get("id-1") is None


def test_clear_on_empty_repository_is_a_no_op(repository: InMemoryExperienceRepository) -> None:
    repository.clear()

    assert repository.count() == 0


def test_add_after_clear_works_again(repository: InMemoryExperienceRepository) -> None:
    repository.add(_make_record("id-1"))
    repository.clear()
    repository.add(_make_record("id-1"))

    assert repository.count() == 1
    assert repository.get("id-1") is not None


def test_count_reflects_current_size(repository: InMemoryExperienceRepository) -> None:
    assert repository.count() == 0
    repository.add(_make_record("id-1"))
    assert repository.count() == 1
    repository.add(_make_record("id-2"))
    assert repository.count() == 2


def test_is_an_experience_repository() -> None:
    repository = InMemoryExperienceRepository()

    assert isinstance(repository, ExperienceRepository)


def test_experience_repository_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        ExperienceRepository()  # type: ignore[abstract]
