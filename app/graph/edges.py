"""Placeholder transition definitions for the ACRF LangGraph execution graph.

This module defines the *shape* of routing: the named outcomes each
conditional branch point can produce, and the path map from each named
outcome to a destination node (or the terminal `END`). It does not decide
which outcome applies for a given state — every conditional edge function
below raises `NotImplementedError`. The decision algorithm (reading
`policy_decision`, `critic_scores`, `safety_status`, `iteration_count`,
etc. to pick an outcome) is deferred to future implementation work.
"""

from enum import Enum

from langgraph.graph import END

from app.graph.nodes import NodeName
from app.state import AgentState


class TerminalCondition(str, Enum):
    """Named conditions under which the graph is expected to terminate.

    These label *why* execution ends; the predicates that detect them
    (e.g. comparing `iteration_count` to `max_iterations`) are not
    implemented here.
    """

    COMPLETED = "completed"
    SAFETY_BLOCKED = "safety_blocked"
    MAX_ITERATIONS_EXCEEDED = "max_iterations_exceeded"
    UNRECOVERABLE_ERROR = "unrecoverable_error"


# --- Router: dispatches to the next stage based on the policy decision ---

ROUTER_OUTCOMES = (
    "retry_worker",
    "evaluate_output",
    "apply_correction",
    "check_safety",
    "finalize",
)

ROUTER_PATH_MAP: dict[str, str] = {
    "retry_worker": NodeName.WORKER.value,
    "evaluate_output": NodeName.CRITIC.value,
    "apply_correction": NodeName.SELF_CORRECTION.value,
    "check_safety": NodeName.SAFETY.value,
    "finalize": NodeName.EVALUATION.value,
}


def route_after_router(state: AgentState) -> str:
    """Select the next node after `router` based on `state.policy_decision`.

    Must return one of the keys in `ROUTER_PATH_MAP`. Not implemented.
    """
    raise NotImplementedError("route_after_router is a placeholder and is not yet implemented.")


# --- Critic: decides whether output proceeds to safety review or correction ---

CRITIC_OUTCOMES = (
    "proceed_to_safety",
    "needs_correction",
)

CRITIC_PATH_MAP: dict[str, str] = {
    "proceed_to_safety": NodeName.SAFETY.value,
    "needs_correction": NodeName.SELF_CORRECTION.value,
}


def route_after_critic(state: AgentState) -> str:
    """Select the next node after `critic` based on `state.critic_scores`
    and `state.aggregated_quality_score` relative to policy thresholds.

    Must return one of the keys in `CRITIC_PATH_MAP`. Not implemented.
    """
    raise NotImplementedError("route_after_critic is a placeholder and is not yet implemented.")


# --- Self-correction: decides whether to retry work or give up ---

SELF_CORRECTION_OUTCOMES = (
    "retry",
    TerminalCondition.MAX_ITERATIONS_EXCEEDED.value,
    TerminalCondition.UNRECOVERABLE_ERROR.value,
)

SELF_CORRECTION_PATH_MAP: dict[str, str] = {
    "retry": NodeName.WORKER.value,
    TerminalCondition.MAX_ITERATIONS_EXCEEDED.value: NodeName.EVALUATION.value,
    TerminalCondition.UNRECOVERABLE_ERROR.value: END,
}


def route_after_self_correction(state: AgentState) -> str:
    """Select the next node after `self_correction`.

    Expected to compare `state.iteration_count` against
    `state.max_iterations` to choose between retrying (`retry`), giving up
    gracefully (`TerminalCondition.MAX_ITERATIONS_EXCEEDED`), or aborting
    (`TerminalCondition.UNRECOVERABLE_ERROR`). Must return one of the keys
    in `SELF_CORRECTION_PATH_MAP`. Not implemented.
    """
    raise NotImplementedError(
        "route_after_self_correction is a placeholder and is not yet implemented."
    )


# --- Safety: decides whether execution may proceed to finalization ---

SAFETY_OUTCOMES = (
    "safe",
    "flagged",
    TerminalCondition.SAFETY_BLOCKED.value,
)

SAFETY_PATH_MAP: dict[str, str] = {
    "safe": NodeName.EVALUATION.value,
    "flagged": NodeName.SELF_CORRECTION.value,
    TerminalCondition.SAFETY_BLOCKED.value: END,
}


def route_after_safety(state: AgentState) -> str:
    """Select the next node after `safety` based on `state.safety_status`.

    Must return one of the keys in `SAFETY_PATH_MAP`. Not implemented.
    """
    raise NotImplementedError("route_after_safety is a placeholder and is not yet implemented.")
