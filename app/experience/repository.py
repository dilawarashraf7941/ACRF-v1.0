"""Storage layer for `ExperienceRecord`s.

Defines an abstract `ExperienceRepository` interface and one concrete,
in-memory-only implementation. No persistence, no database, no ChromaDB,
no SQLite — this module only stores records for the lifetime of the
Python process. The abstract interface exists specifically so a future
SQLite/ChromaDB/PostgreSQL-backed implementation can be substituted
without requiring any change to `ExperienceRecorder` or to any other code
that depends on `ExperienceRepository`.
"""

from abc import ABC, abstractmethod

from app.experience.models import ExperienceRecord


class ExperienceRepository(ABC):
    """Abstract interface for storing and retrieving `ExperienceRecord`s.

    Any concrete implementation (in-memory, SQLite, ChromaDB, PostgreSQL,
    ...) must provide these five operations. Code that depends on this
    interface — including `ExperienceRecorder` — never needs to know which
    concrete implementation it is talking to.
    """

    @abstractmethod
    def add(self, record: ExperienceRecord) -> None:
        """Store `record`.

        Args:
            record: The experience to store.

        Raises:
            ValueError: If a record with the same `experience_id` already exists.
        """
        raise NotImplementedError

    @abstractmethod
    def get(self, experience_id: str) -> ExperienceRecord | None:
        """Retrieve a single record by id.

        Args:
            experience_id: The identifier to look up.

        Returns:
            The matching `ExperienceRecord`, or `None` if no record with
            that id has been stored.
        """
        raise NotImplementedError

    @abstractmethod
    def list(self) -> list[ExperienceRecord]:
        """Return every stored record.

        Returns:
            All stored `ExperienceRecord`s. Implementations should define
            a stable, documented order (this one returns insertion order).
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Remove every stored record."""
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored records."""
        raise NotImplementedError


class InMemoryExperienceRepository(ExperienceRepository):
    """An in-memory-only `ExperienceRepository`.

    Records live only in a plain Python dict for the lifetime of this
    instance; nothing is written to disk, to a database, or to ChromaDB.
    Records are keyed by `experience_id`, so `add` rejects a second record
    with an id that is already present rather than silently overwriting
    it.
    """

    def __init__(self) -> None:
        """Create an empty repository."""
        self._records: dict[str, ExperienceRecord] = {}

    def add(self, record: ExperienceRecord) -> None:
        """Store `record`, keyed by its `experience_id`.

        Args:
            record: The experience to store.

        Raises:
            ValueError: If a record with the same `experience_id` has
                already been added to this repository.
        """
        if record.experience_id in self._records:
            raise ValueError(
                f"An experience with id {record.experience_id!r} already exists in this repository."
            )
        self._records[record.experience_id] = record

    def get(self, experience_id: str) -> ExperienceRecord | None:
        """Retrieve a stored record by id, or `None` if not found."""
        return self._records.get(experience_id)

    def list(self) -> list[ExperienceRecord]:
        """Return every stored record, in insertion order."""
        return list(self._records.values())

    def clear(self) -> None:
        """Remove every stored record."""
        self._records.clear()

    def count(self) -> int:
        """Return the number of stored records."""
        return len(self._records)


DEFAULT_EXPERIENCE_REPOSITORY: ExperienceRepository = InMemoryExperienceRepository()
"""A process-wide, shared `InMemoryExperienceRepository` instance.

`evaluation_node` (see `app/graph/nodes.py`) is a plain function invoked
by LangGraph with only `(state)` — there is no constructor or call site
available to inject a repository into it from outside. This module-level
singleton is the practical mechanism by which `evaluation_node` obtains a
repository that persists across calls within a process. Tests should
construct and inject their own `InMemoryExperienceRepository` instance
instead of relying on this shared singleton, to stay isolated from other
tests and from other code sharing the same process.
"""
