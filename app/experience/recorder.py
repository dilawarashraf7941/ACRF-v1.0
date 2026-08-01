"""`ExperienceRecorder`: builds `ExperienceRecord`s from `AgentState`.

Reads only `AgentState`. No business logic, no learning, no scoring, no
routing — every value copied into the resulting `ExperienceRecord` is
either transcribed directly from an existing `AgentState` field, or
derived via a purely mechanical operation (a timestamp subtraction, a sum
of already-present numbers, a dict-key lookup). No decision is made about
whether the execution was "good," and nothing here influences routing,
critics, or policy.
"""

import hashlib

from app.experience.models import ExperienceRecord
from app.experience.repository import ExperienceRepository
from app.state import AgentState


def _build_experience_id(session_id: str, task_id: str, iterations: int) -> str:
    """Deterministically derive a stable experience id.

    The same `(session_id, task_id, iterations)` always produces the same
    id (a SHA-256 hex digest), and different inputs produce different ids
    with cryptographic collision resistance. No randomness and no
    timestamp are involved, so the id is a pure function of its inputs.

    Args:
        session_id: `state.session_id`.
        task_id: `state.task_id`.
        iterations: `state.iteration_count`.

    Returns:
        A 64-character hex digest identifying this `(session_id, task_id,
        iterations)` triple.
    """
    seed = f"{session_id}|{task_id}|{iterations}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _compute_latency(state: AgentState) -> float | None:
    """Compute elapsed seconds from `state.execution_metadata`'s timestamps.

    Args:
        state: The current agent state.

    Returns:
        `(updated_at - created_at).total_seconds()`, or `None` if either
        timestamp is unavailable.
    """
    metadata = state.execution_metadata
    if metadata.created_at is None or metadata.updated_at is None:
        return None
    return (metadata.updated_at - metadata.created_at).total_seconds()


def _compute_estimated_cost(state: AgentState) -> float:
    """Sum worker token usage as a placeholder cost proxy.

    No pricing model exists in this framework; this is a raw sum of an
    already-present numeric field, not a cost calculation.

    Args:
        state: The current agent state.

    Returns:
        The sum of `token_usage` (an additive `WorkerOutput` field, see
        `worker_node`) across `state.worker_outputs`, treating a missing
        or non-numeric value as `0`.
    """
    total = 0.0
    for worker_output in state.worker_outputs:
        token_usage = getattr(worker_output, "token_usage", 0)
        if isinstance(token_usage, (int, float)):
            total += float(token_usage)
    return total


def _build_state_features(state: AgentState) -> dict[str, object]:
    """Snapshot the AgentState signals relevant to a future learner.

    Pure transcription: every value here already exists on `state`; none
    of it is computed, scored, or interpreted.

    Args:
        state: The current agent state.

    Returns:
        A plain dict summarizing `task_type`, iteration counts, the
        latest planner output, and the full list of error features.
    """
    return {
        "task_type": state.task_type,
        "iteration_count": state.iteration_count,
        "max_iterations": state.max_iterations,
        "worker_output_count": len(state.worker_outputs),
        "error_feature_count": len(state.error_features),
        "error_features": [feature.model_dump(mode="json") for feature in state.error_features],
        "planner_output": (
            state.planner_output.model_dump(mode="json")
            if state.planner_output is not None
            else None
        ),
    }


def _build_memory_usage(state: AgentState) -> dict[str, object]:
    """Snapshot the memory-subsystem-related signals present on AgentState.

    Args:
        state: The current agent state.

    Returns:
        A plain dict with the count of retrieved memories and the set of
        keys currently present in `state.memory_context`.
    """
    return {
        "retrieved_memories_count": len(state.retrieved_memories),
        "memory_context_keys": sorted(state.memory_context.keys()),
    }


def _extract_correction_decision(state: AgentState) -> dict[str, object] | None:
    """Read the correction policy's decision out of `state.memory_context`, if present.

    Args:
        state: The current agent state.

    Returns:
        `state.memory_context["correction_policy"]["decision"]` if both
        keys are present and shaped as expected, else `None`.
    """
    correction_policy_entry = state.memory_context.get("correction_policy")
    if not isinstance(correction_policy_entry, dict):
        return None
    decision = correction_policy_entry.get("decision")
    return decision if isinstance(decision, dict) else None


class ExperienceRecorder:
    """Builds `ExperienceRecord`s from `AgentState`, reading only `AgentState`.

    Optionally accepts an `ExperienceRepository` (dependency injection),
    in which case every record built by `record` is also stored into it.
    Without a repository, `record` only builds and returns the record —
    it is never required to have storage side effects.
    """

    def __init__(self, repository: ExperienceRepository | None = None) -> None:
        """Create a recorder, optionally wired to a repository.

        Args:
            repository: An `ExperienceRepository` to store every recorded
                experience into, or `None` to only build records without
                storing them.
        """
        self._repository = repository

    def record(self, state: AgentState) -> ExperienceRecord:
        """Build an `ExperienceRecord` from `state`.

        If this recorder was constructed with a repository, the built
        record is also added to it before being returned.

        Args:
            state: The current agent state. Read-only: no field of
                `state` is modified.

        Returns:
            The newly built `ExperienceRecord`.
        """
        experience_id = _build_experience_id(
            state.session_id, state.task_id, state.iteration_count
        )
        experience = ExperienceRecord(
            experience_id=experience_id,
            session_id=state.session_id,
            task_id=state.task_id,
            timestamp=state.execution_metadata.updated_at,
            state_features=_build_state_features(state),
            selected_critics=list(state.selected_critics),
            critic_scores=dict(state.critic_scores),
            aggregated_quality_score=state.aggregated_quality_score,
            correction_decision=_extract_correction_decision(state),
            iterations=state.iteration_count,
            final_response=state.final_response,
            execution_status=state.execution_status.value,
            latency=_compute_latency(state),
            estimated_cost=_compute_estimated_cost(state),
            memory_usage=_build_memory_usage(state),
            metadata={"recorded_by": "ExperienceRecorder", "source_node": "evaluation_node"},
        )

        if self._repository is not None:
            self._repository.add(experience)

        return experience
