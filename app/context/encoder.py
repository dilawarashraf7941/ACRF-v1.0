"""`ContextEncoder`: builds a `ContextVector` from an `AgentState` (and,
optionally, a matching `ExperienceRecord`).

No reinforcement learning, no contextual bandits, no policy optimization,
and no learning of any kind — every feature below is either copied
directly, a fixed categorical-to-numeric mapping, or a simple derived
ratio/aggregate computed with a fixed formula. Nothing is fitted from a
dataset or batch of prior contexts (that would be a form of learning a
distribution's statistics); every constant here is chosen at
implementation time, matching the same "fixed, hand-authored heuristic"
convention used throughout `app/policy_engine`, `app/correction_policy`,
and `app/reward`.

Nine of the produced features (`uncertainty`, `risk`, `task_complexity`,
`memory_relevance`, `requires_self_correction`, `requires_meta_critic`,
`is_code_output`, `iteration_pressure`, `attempt_pressure`) mirror
`app/policy_engine/scorer.py`'s `HeuristicPolicyScorer.extract_features`
exactly (same formulas, same thresholds), so a `HeuristicPolicy` (see
`app/policy`) built on top of this module's `ContextVector` produces
identical scores to the pre-refactor, directly-`AgentState`-based scorer.
The extraction functions are intentionally duplicated here, not
imported, so `app/context` has no dependency on `app/policy_engine` —
the same "small modules stay independent" convention already used by
`app/correction_policy` (see its README for the same rationale).
"""

import hashlib
from typing import Any

from app.context.models import ContextVector
from app.experience import ExperienceRecord
from app.state import AgentState, ErrorFeature, PlannerOutput, WorkerOutput

SAFETY_STATUS_CODES: dict[str, float] = {
    "unknown": 0.0,
    "safe": 1.0,
    "flagged": 2.0,
    "blocked": 3.0,
}
"""Fixed ordinal encoding for `AgentState.safety_status.value`."""

EXECUTION_STATUS_CODES: dict[str, float] = {
    "pending": 0.0,
    "running": 1.0,
    "paused": 2.0,
    "completed": 3.0,
    "failed": 4.0,
    "cancelled": 5.0,
}
"""Fixed ordinal encoding for `AgentState.execution_status.value`."""

UNRECOGNIZED_STATUS_CODE = -1.0
"""Code used for a status string not present in the fixed encoding tables above
(e.g. a future status value), so encoding never raises on unrecognized input."""

_RISK_LEVEL_SCORES: dict[str, float] = {
    "low": 0.0,
    "medium": 0.4,
    "high": 0.75,
    "critical": 1.0,
}
"""Mirrors `app.policy_engine.scorer._RISK_LEVEL_SCORES` exactly."""

_TASK_COMPLEXITY_SCORES: dict[str, float] = {
    "trivial": 0.0,
    "simple": 0.25,
    "moderate": 0.5,
    "complex": 0.75,
    "very_complex": 1.0,
}
"""Mirrors `app.policy_engine.scorer._TASK_COMPLEXITY_SCORES` exactly."""

_PLAN_COMPLEXITY_STEP_CAP = 5
"""Mirrors `app.policy_engine.scorer._PLAN_COMPLEXITY_STEP_CAP` exactly."""

_ITERATION_PRESSURE_CAP = 5
"""Mirrors `app.policy_engine.scorer._ITERATION_PRESSURE_CAP` exactly."""

_ATTEMPT_PRESSURE_CAP = 4
"""Mirrors `app.policy_engine.scorer._ATTEMPT_PRESSURE_CAP` exactly."""


def _clamp(value: float) -> float:
    """Clamp `value` into the closed interval `[0.0, 1.0]`."""
    return max(0.0, min(1.0, value))


def _latest_error_feature(error_features: list[ErrorFeature]) -> ErrorFeature | None:
    """Return the most recently extracted `ErrorFeature`, or `None` if there is none."""
    return error_features[-1] if error_features else None


def _latest_profile(error_features: list[ErrorFeature]) -> dict[str, Any] | None:
    """Return the nested `ErrorFeatureProfile` dump from the latest error feature's metadata.

    Mirrors `app.policy_engine.scorer._latest_profile` exactly.
    """
    latest = _latest_error_feature(error_features)
    if latest is None:
        return None
    profile = latest.metadata.get("profile")
    return profile if isinstance(profile, dict) else None


def _extract_uncertainty(error_features: list[ErrorFeature]) -> float:
    """Derive an uncertainty score from the latest error feature's confidence.

    Mirrors `app.policy_engine.scorer._extract_uncertainty` exactly.
    """
    latest = _latest_error_feature(error_features)
    if latest is None:
        return 0.0

    confidence = latest.metadata.get("confidence")
    if not isinstance(confidence, (int, float)):
        profile = _latest_profile(error_features)
        confidence = profile.get("confidence_score") if profile else None
    if not isinstance(confidence, (int, float)):
        confidence = 1.0

    return _clamp(1.0 - float(confidence))


def _extract_risk(error_features: list[ErrorFeature]) -> float:
    """Derive a numeric risk score from the latest error feature's risk level.

    Mirrors `app.policy_engine.scorer._extract_risk` exactly.
    """
    latest = _latest_error_feature(error_features)
    if latest is None:
        return 0.0

    risk_level = latest.metadata.get("risk_level")
    if isinstance(risk_level, str):
        return _RISK_LEVEL_SCORES.get(risk_level.lower(), 0.0)
    return 0.0


def _extract_error_task_complexity(error_features: list[ErrorFeature]) -> float:
    """Derive a task-complexity score from the latest error feature's profile.

    Mirrors `app.policy_engine.scorer._extract_error_task_complexity` exactly.
    """
    profile = _latest_profile(error_features)
    if profile is None:
        return 0.0

    complexity = profile.get("task_complexity")
    if isinstance(complexity, str):
        return _TASK_COMPLEXITY_SCORES.get(complexity.lower(), 0.0)
    return 0.0


def _extract_plan_complexity(planner_output: PlannerOutput | None) -> float:
    """Derive a task-complexity signal from the planner's decomposition length.

    Mirrors `app.policy_engine.scorer._extract_plan_complexity` exactly.
    """
    if planner_output is None:
        return 0.0
    decomposition = getattr(planner_output, "decomposition", None)
    if not isinstance(decomposition, list):
        return 0.0
    return _clamp(len(decomposition) / _PLAN_COMPLEXITY_STEP_CAP)


def _extract_requires_flag(error_features: list[ErrorFeature], key: str) -> bool:
    """Read a boolean flag (e.g. `requires_meta_critic`) from the latest error feature's profile.

    Mirrors `app.policy_engine.scorer._extract_requires_flag` exactly.
    """
    profile = _latest_profile(error_features)
    if profile is None:
        return False
    return bool(profile.get(key))


def _extract_profile_memory_relevance(error_features: list[ErrorFeature]) -> float:
    """Read `memory_relevance` from the latest error feature's profile.

    Mirrors `app.policy_engine.scorer._extract_profile_memory_relevance` exactly.
    """
    profile = _latest_profile(error_features)
    if profile is None:
        return 0.0
    value = profile.get("memory_relevance")
    if isinstance(value, (int, float)):
        return _clamp(float(value))
    return 0.0


def _extract_context_memory_relevance(memory_context: dict[str, Any]) -> float:
    """Read an explicit `memory_relevance` signal from `state.memory_context`, if present.

    Mirrors `app.policy_engine.scorer._extract_context_memory_relevance` exactly.
    """
    value = memory_context.get("memory_relevance")
    if isinstance(value, (int, float)):
        return _clamp(float(value))
    return 0.0


def _extract_is_code_output(error_features: list[ErrorFeature]) -> bool:
    """Read whether the latest error feature classified the output as code.

    Mirrors `app.policy_engine.scorer._extract_is_code_output` exactly.
    """
    latest = _latest_error_feature(error_features)
    if latest is None:
        return False
    return latest.metadata.get("output_type") == "code"


def _extract_iteration_pressure(iteration_count: int) -> float:
    """Normalize `iteration_count` into an escalating pressure signal.

    Mirrors `app.policy_engine.scorer._extract_iteration_pressure` exactly.
    """
    if iteration_count <= 0:
        return 0.0
    return _clamp(iteration_count / _ITERATION_PRESSURE_CAP)


def _extract_attempt_pressure(worker_outputs: list[WorkerOutput]) -> float:
    """Normalize the number of *extra* worker attempts into a pressure signal.

    Mirrors `app.policy_engine.scorer._extract_attempt_pressure` exactly.
    """
    extra_attempts = len(worker_outputs) - 1
    if extra_attempts <= 0:
        return 0.0
    return _clamp(extra_attempts / _ATTEMPT_PRESSURE_CAP)


def _resolve_task_type(state: AgentState) -> str | None:
    """Determine the task type used for encoding.

    Mirrors the same resolution order used elsewhere in ACRF (prefer
    `state.task_type`, fall back to `state.planner_output.task_type`);
    duplicated locally rather than imported, since `app/context` must not
    depend on `app/graph`.

    Args:
        state: The current agent state.

    Returns:
        The resolved task type, or `None` if neither source has one.
    """
    if state.task_type is not None:
        return state.task_type
    if state.planner_output is not None:
        return getattr(state.planner_output, "task_type", None)
    return None


def _build_context_id(session_id: str, task_id: str, iterations: int) -> str:
    """Deterministically derive a stable context id.

    Salted with a fixed `"context|"` prefix so it never collides with
    `ExperienceRecord.experience_id`/`ExecutionMetrics.execution_id`
    (which hash the same triple without a salt), even though both are
    derived from the same `(session_id, task_id, iterations)` triple.

    Args:
        session_id: `state.session_id`.
        task_id: `state.task_id`.
        iterations: `state.iteration_count`.

    Returns:
        A 64-character hex digest identifying this context.
    """
    seed = f"context|{session_id}|{task_id}|{iterations}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _status_code(value: str, table: dict[str, float]) -> float:
    """Look up a fixed ordinal code for `value`, defaulting gracefully.

    Args:
        value: The status string to encode (e.g. `state.safety_status.value`).
        table: The fixed encoding table to use.

    Returns:
        `table[value]` if present, else `UNRECOGNIZED_STATUS_CODE`.
    """
    return table.get(value, UNRECOGNIZED_STATUS_CODE)


class ContextEncoder:
    """Deterministically encodes an `AgentState` into a numeric `ContextVector`.

    `state` is the sole source of every entry in the returned vector's
    `features` — the pre-decision observation a future policy would
    actually have available. An optional `experience` is read *only* for
    supplementary, clearly-labeled outcome-derived data stored under
    `metadata["experience_derived"]`, kept out of `features` entirely so
    it can never be mistaken for information available before an outcome
    is known.
    """

    def encode(
        self, state: AgentState, experience: ExperienceRecord | None = None
    ) -> ContextVector:
        """Build a `ContextVector` from `state` (and, optionally, `experience`).

        Args:
            state: The current agent state. Read-only: no field of
                `state` is modified. The sole source of `features`.
            experience: An optional matching `ExperienceRecord`, used only
                to populate `metadata["experience_derived"]`.

        Returns:
            The newly built, unnormalized `ContextVector`
            (`normalized=False`).
        """
        features = self._encode_features(state)

        metadata: dict[str, object] = {"source": "AgentState"}
        if experience is not None:
            metadata["source"] = "AgentState+ExperienceRecord"
            metadata["experience_derived"] = self._encode_experience_derived(experience)

        timestamp = (
            experience.timestamp if experience is not None else state.execution_metadata.updated_at
        )

        return ContextVector(
            context_id=_build_context_id(state.session_id, state.task_id, state.iteration_count),
            source_execution_id=experience.experience_id if experience is not None else None,
            features=features,
            feature_order=list(features.keys()),
            normalized=False,
            normalization_strategy=None,
            timestamp=timestamp,
            metadata=metadata,
        )

    def _encode_features(self, state: AgentState) -> dict[str, float]:
        """Encode the pre-decision-safe numeric features of `state`.

        Args:
            state: The current agent state.

        Returns:
            An ordered dict of named numeric features.
        """
        critic_scores = list(state.critic_scores.values())
        max_iterations = float(state.max_iterations)
        iteration_ratio = (
            float(state.iteration_count) / max_iterations if max_iterations > 0 else 0.0
        )
        aggregated_quality_score = state.aggregated_quality_score
        task_type = _resolve_task_type(state)
        error_features = state.error_features

        return {
            "iteration_count": float(state.iteration_count),
            "max_iterations": max_iterations,
            "iteration_ratio": max(0.0, min(1.0, iteration_ratio)),
            "error_feature_count": float(len(state.error_features)),
            "worker_output_count": float(len(state.worker_outputs)),
            "critic_score_count": float(len(state.critic_scores)),
            "selected_critics_count": float(len(state.selected_critics)),
            "retrieved_memories_count": float(len(state.retrieved_memories)),
            "correction_history_count": float(len(state.correction_history)),
            "aggregated_quality_score": (
                aggregated_quality_score if aggregated_quality_score is not None else 0.0
            ),
            "has_aggregated_quality_score": 1.0 if aggregated_quality_score is not None else 0.0,
            "safety_status_code": _status_code(state.safety_status.value, SAFETY_STATUS_CODES),
            "execution_status_code": _status_code(
                state.execution_status.value, EXECUTION_STATUS_CODES
            ),
            "is_code_task": 1.0 if task_type == "code" else 0.0,
            "has_task_type": 1.0 if task_type is not None else 0.0,
            "average_critic_score": (
                sum(critic_scores) / len(critic_scores) if critic_scores else 0.0
            ),
            "max_critic_score": max(critic_scores) if critic_scores else 0.0,
            "min_critic_score": min(critic_scores) if critic_scores else 0.0,
            # The following nine mirror HeuristicPolicyScorer.extract_features
            # exactly (see module docstring), so HeuristicPolicy can be built
            # purely on top of ContextVector.features with identical behavior.
            "uncertainty": _extract_uncertainty(error_features),
            "risk": _extract_risk(error_features),
            "task_complexity": max(
                _extract_error_task_complexity(error_features),
                _extract_plan_complexity(state.planner_output),
            ),
            "memory_relevance": max(
                _extract_profile_memory_relevance(error_features),
                _extract_context_memory_relevance(state.memory_context),
            ),
            "requires_self_correction": (
                1.0 if _extract_requires_flag(error_features, "requires_self_correction") else 0.0
            ),
            "requires_meta_critic": (
                1.0 if _extract_requires_flag(error_features, "requires_meta_critic") else 0.0
            ),
            "is_code_output": 1.0 if _extract_is_code_output(error_features) else 0.0,
            "iteration_pressure": _extract_iteration_pressure(state.iteration_count),
            "attempt_pressure": _extract_attempt_pressure(state.worker_outputs),
        }

    def _encode_experience_derived(self, experience: ExperienceRecord) -> dict[str, float]:
        """Encode supplementary, outcome-derived numeric features from `experience`.

        Deliberately kept out of `features`: `latency` and `estimated_cost`
        are only known *after* an execution completes, so including them
        in the primary context would leak post-decision information into
        what is meant to represent the pre-decision observation.

        Args:
            experience: The `ExperienceRecord` to read supplementary data from.

        Returns:
            A dict of outcome-derived numeric features.
        """
        estimated_cost = experience.estimated_cost
        return {
            "latency": experience.latency if experience.latency is not None else 0.0,
            "has_latency": 1.0 if experience.latency is not None else 0.0,
            "estimated_cost": estimated_cost if estimated_cost is not None else 0.0,
            "has_estimated_cost": 1.0 if estimated_cost is not None else 0.0,
        }
