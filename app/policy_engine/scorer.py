"""Deterministic, feature-based critic scoring for the Adaptive Policy Engine.

`HeuristicPolicyScorer` replaces the constant placeholder policy score
(`PlaceholderPolicyEngine.score`, see `app/policies/engine.py`, which
always returns `0.0`) with a real, deterministic scoring function: a
fixed, hand-authored weighted sum over features extracted from
`AgentState`. There is no reinforcement learning, no neural network, no
LLM or API call, and no randomness anywhere in this module — identical
inputs always produce identical scores, and every weight below is a
literal constant chosen at implementation time, not learned or fitted.

Only `AgentState.error_features`, `memory_context`, `iteration_count`,
`planner_output`, and `worker_outputs` are read. No other `AgentState`
field (notably not `task_type`, `selected_critics`, or `policy_decision`)
is consulted or written by this module.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from app.state import AgentState, ErrorFeature, PlannerOutput, WorkerOutput

_RISK_LEVEL_SCORES: dict[str, float] = {
    "low": 0.0,
    "medium": 0.4,
    "high": 0.75,
    "critical": 1.0,
}
"""Maps an `ErrorFeatureProfile`-style risk level string to a numeric 0.0-1.0 score."""

_TASK_COMPLEXITY_SCORES: dict[str, float] = {
    "trivial": 0.0,
    "simple": 0.25,
    "moderate": 0.5,
    "complex": 0.75,
    "very_complex": 1.0,
}
"""Maps an `ErrorFeatureProfile`-style task complexity string to a numeric 0.0-1.0 score."""

_PLAN_COMPLEXITY_STEP_CAP = 5
"""Number of `planner_output.decomposition` steps at which plan-complexity saturates to 1.0."""

_ITERATION_PRESSURE_CAP = 5
"""Iteration count at which `iteration_pressure` saturates to 1.0."""

_ATTEMPT_PRESSURE_CAP = 4
"""Number of *extra* worker outputs (beyond the first) at which `attempt_pressure` saturates."""

_CRITIC_WEIGHTS: dict[str, dict[str, float]] = {
    "LogicCritic": {
        "base": 0.10,
        "uncertainty": 0.35,
        "risk": 0.25,
        "task_complexity": 0.30,
    },
    "CodeCritic": {
        "base": 0.10,
        "is_code_output": 0.50,
        "task_complexity": 0.25,
        "risk": 0.15,
    },
    "FactCritic": {
        "base": 0.10,
        "memory_relevance": 0.40,
        "uncertainty": 0.30,
        "risk": 0.20,
    },
    "MetaCritic": {
        "base": 0.0,
        "requires_meta_critic": 0.35,
        "iteration_pressure": 0.25,
        "requires_self_correction": 0.15,
        "attempt_pressure": 0.15,
        "risk": 0.10,
    },
}
"""Per-critic feature weights. Each critic's weights (including `base`) sum to 1.0, so with
every feature in `[0, 1]`, `score_critic` always returns a value in `[0, 1]`. These are the
hand-authored heuristic parameters this module implements — fixed constants, not learned."""

_DEFAULT_WEIGHTS: dict[str, float] = {
    "base": 0.0,
    "uncertainty": 0.25,
    "risk": 0.25,
    "task_complexity": 0.20,
    "memory_relevance": 0.10,
    "requires_meta_critic": 0.10,
    "requires_self_correction": 0.10,
}
"""Fallback weights used for any candidate critic name not present in `_CRITIC_WEIGHTS`."""


@dataclass(frozen=True)
class StateFeatures:
    """The fixed set of deterministic features extracted from `AgentState` for scoring.

    Every field is normalized to `[0.0, 1.0]` (or is a plain `bool`), so
    the weighted sums in `score_critic` are easy to reason about and
    always land in `[0.0, 1.0]`.
    """

    uncertainty: float
    risk: float
    task_complexity: float
    memory_relevance: float
    requires_self_correction: bool
    requires_meta_critic: bool
    is_code_output: bool
    iteration_pressure: float
    attempt_pressure: float


def _clamp(value: float) -> float:
    """Clamp `value` into the closed interval `[0.0, 1.0]`."""
    return max(0.0, min(1.0, value))


def _latest_error_feature(error_features: list[ErrorFeature]) -> ErrorFeature | None:
    """Return the most recently extracted `ErrorFeature`, or `None` if there is none."""
    return error_features[-1] if error_features else None


def _latest_profile(error_features: list[ErrorFeature]) -> dict[str, Any] | None:
    """Return the nested `ErrorFeatureProfile` dump from the latest error feature's metadata.

    Returns `None` if there is no error feature or no profile. See
    `app/graph/nodes.py::error_feature_extractor_node`, which stores a
    full `ErrorFeatureProfile.model_dump()` under `metadata["profile"]`.
    """
    latest = _latest_error_feature(error_features)
    if latest is None:
        return None
    profile = latest.metadata.get("profile")
    return profile if isinstance(profile, dict) else None


def _extract_uncertainty(error_features: list[ErrorFeature]) -> float:
    """Derive an uncertainty score from the latest error feature's confidence.

    Prefers `metadata["confidence"]`, falling back to
    `metadata["profile"]["confidence_score"]`, and defaults to full
    confidence (`uncertainty=0.0`) when no error feature or confidence
    value is available.

    Args:
        error_features: `state.error_features`.

    Returns:
        `1.0 - confidence`, clamped to `[0.0, 1.0]`.
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

    Args:
        error_features: `state.error_features`.

    Returns:
        A value from `_RISK_LEVEL_SCORES`, or `0.0` if unavailable/unrecognized.
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

    Args:
        error_features: `state.error_features`.

    Returns:
        A value from `_TASK_COMPLEXITY_SCORES`, or `0.0` if unavailable/unrecognized.
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

    Args:
        planner_output: `state.planner_output`.

    Returns:
        `len(decomposition) / _PLAN_COMPLEXITY_STEP_CAP`, clamped to `[0.0, 1.0]`;
        `0.0` if `planner_output` is `None` or has no decomposition.
    """
    if planner_output is None:
        return 0.0
    decomposition = getattr(planner_output, "decomposition", None)
    if not isinstance(decomposition, list):
        return 0.0
    return _clamp(len(decomposition) / _PLAN_COMPLEXITY_STEP_CAP)


def _extract_requires_flag(error_features: list[ErrorFeature], key: str) -> bool:
    """Read a boolean flag (e.g. `requires_meta_critic`) from the latest error feature's profile.

    Args:
        error_features: `state.error_features`.
        key: The profile field name to read.

    Returns:
        `True` if the profile has a truthy value for `key`, else `False`.
    """
    profile = _latest_profile(error_features)
    if profile is None:
        return False
    return bool(profile.get(key))


def _extract_profile_memory_relevance(error_features: list[ErrorFeature]) -> float:
    """Read `memory_relevance` from the latest error feature's profile.

    Args:
        error_features: `state.error_features`.

    Returns:
        The profile's `memory_relevance`, clamped to `[0.0, 1.0]`; `0.0` if unavailable.
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

    This is a forward-compatible hook for a future memory subsystem (see
    `app/memory/`): any component that has assessed how relevant retrieved
    memory is for the current task can publish a `memory_relevance` float
    into `state.memory_context`, and this scorer will pick it up.

    Args:
        memory_context: `state.memory_context`.

    Returns:
        `memory_context["memory_relevance"]`, clamped to `[0.0, 1.0]`; `0.0` if absent/invalid.
    """
    value = memory_context.get("memory_relevance")
    if isinstance(value, (int, float)):
        return _clamp(float(value))
    return 0.0


def _extract_is_code_output(error_features: list[ErrorFeature]) -> bool:
    """Read whether the latest error feature classified the output as code.

    Args:
        error_features: `state.error_features`.

    Returns:
        `True` if `metadata["output_type"] == "code"`, else `False`.
    """
    latest = _latest_error_feature(error_features)
    if latest is None:
        return False
    return latest.metadata.get("output_type") == "code"


def _extract_iteration_pressure(iteration_count: int) -> float:
    """Normalize `iteration_count` into an escalating pressure signal.

    Args:
        iteration_count: `state.iteration_count`.

    Returns:
        `iteration_count / _ITERATION_PRESSURE_CAP`, clamped to `[0.0, 1.0]`.
    """
    if iteration_count <= 0:
        return 0.0
    return _clamp(iteration_count / _ITERATION_PRESSURE_CAP)


def _extract_attempt_pressure(worker_outputs: list[WorkerOutput]) -> float:
    """Normalize the number of *extra* worker attempts into a pressure signal.

    The first worker output is the initial attempt, not a retry, so only
    subsequent entries (e.g. appended by `self_correction_node`)
    contribute pressure.

    Args:
        worker_outputs: `state.worker_outputs`.

    Returns:
        `(len(worker_outputs) - 1) / _ATTEMPT_PRESSURE_CAP`, clamped to `[0.0, 1.0]`.
    """
    extra_attempts = len(worker_outputs) - 1
    if extra_attempts <= 0:
        return 0.0
    return _clamp(extra_attempts / _ATTEMPT_PRESSURE_CAP)


class HeuristicPolicyScorer:
    """Deterministic, feature-based scorer for candidate critics.

    Reads only `state.error_features`, `state.memory_context`,
    `state.iteration_count`, `state.planner_output`, and
    `state.worker_outputs`, and produces one numeric score in `[0.0, 1.0]`
    per candidate critic name via a fixed weighted sum of features (see
    `_CRITIC_WEIGHTS`). No learning, no LLM/API calls, no randomness:
    identical `AgentState` values always produce identical scores.
    """

    def score(self, state: AgentState, candidate_critics: Iterable[str]) -> dict[str, float]:
        """Score every candidate critic against the features in `state`.

        Args:
            state: The current agent state.
            candidate_critics: The critic identifiers to score.

        Returns:
            A mapping of critic identifier to score in `[0.0, 1.0]`, in
            the same order as `candidate_critics`.
        """
        features = self.extract_features(state)
        return {
            critic_name: self.score_critic(critic_name, features)
            for critic_name in candidate_critics
        }

    def extract_features(self, state: AgentState) -> StateFeatures:
        """Extract the deterministic `StateFeatures` used for scoring.

        Exposed separately from `score` so callers (and tests) can inspect
        the intermediate feature values directly.

        Args:
            state: The current agent state.

        Returns:
            The `StateFeatures` derived from `state`.
        """
        error_features = state.error_features
        return StateFeatures(
            uncertainty=_extract_uncertainty(error_features),
            risk=_extract_risk(error_features),
            task_complexity=max(
                _extract_error_task_complexity(error_features),
                _extract_plan_complexity(state.planner_output),
            ),
            memory_relevance=max(
                _extract_profile_memory_relevance(error_features),
                _extract_context_memory_relevance(state.memory_context),
            ),
            requires_self_correction=_extract_requires_flag(
                error_features, "requires_self_correction"
            ),
            requires_meta_critic=_extract_requires_flag(error_features, "requires_meta_critic"),
            is_code_output=_extract_is_code_output(error_features),
            iteration_pressure=_extract_iteration_pressure(state.iteration_count),
            attempt_pressure=_extract_attempt_pressure(state.worker_outputs),
        )

    def score_critic(self, critic_name: str, features: StateFeatures) -> float:
        """Compute a single critic's score as a fixed weighted sum of `features`.

        Public (not a private helper) specifically so other modules — e.g.
        `app/policy/heuristic_policy.py`'s `HeuristicPolicy` — can reuse
        this exact weighted-sum computation against a `StateFeatures`
        instance built from a source other than a live `AgentState`,
        without duplicating the weight table or the formula.

        Args:
            critic_name: The critic identifier being scored.
            features: The `StateFeatures` to score against.

        Returns:
            The weighted sum, clamped to `[0.0, 1.0]` and rounded to 6
            decimal places for stable, reproducible comparisons.
        """
        weights = _CRITIC_WEIGHTS.get(critic_name, _DEFAULT_WEIGHTS)
        score = weights.get("base", 0.0)
        score += weights.get("uncertainty", 0.0) * features.uncertainty
        score += weights.get("risk", 0.0) * features.risk
        score += weights.get("task_complexity", 0.0) * features.task_complexity
        score += weights.get("memory_relevance", 0.0) * features.memory_relevance
        score += weights.get("requires_self_correction", 0.0) * float(
            features.requires_self_correction
        )
        score += weights.get("requires_meta_critic", 0.0) * float(features.requires_meta_critic)
        score += weights.get("is_code_output", 0.0) * float(features.is_code_output)
        score += weights.get("iteration_pressure", 0.0) * features.iteration_pressure
        score += weights.get("attempt_pressure", 0.0) * features.attempt_pressure
        return round(_clamp(score), 6)
