"""Modular, deterministic heuristic rules for the Correction Decision Policy.

Each rule is a small, pure function: it takes primitive inputs (not
`AgentState` directly), applies one fixed, deterministic condition, and
returns a `RuleResult`. This keeps every rule independently constructible
and testable without needing to build a full `AgentState`, and keeps
`CorrectionDecisionEngine` (see `app/correction_policy/decision.py`)
free of the individual rule conditions themselves.

No reinforcement learning, no learned thresholds, no LLM calls — every
threshold below is a fixed constant chosen at implementation time.
"""

from dataclasses import dataclass
from typing import Any, Literal

Signal = Literal["correct", "no_correct", "neutral"]
"""What a triggered rule advocates: require correction, forbid it, or (when not
triggered) express no opinion."""

QUALITY_THRESHOLD = 0.7
"""Below this, `aggregated_quality_score` or a critic score is considered low quality."""

META_CRITIC_ESCALATION_THRESHOLD = 0.7
"""Above this, `MetaCritic`'s score is treated as an escalation signal demanding correction."""

LOW_MEMORY_RELEVANCE_THRESHOLD = 0.2
"""Below this, an explicit `memory_context["memory_relevance"]` signal is considered too
low to trust the current grounding, and correction is favored."""


@dataclass(frozen=True)
class RuleResult:
    """The outcome of evaluating a single deterministic rule."""

    rule_name: str
    """Stable identifier for this rule, used in `CorrectionDecision.triggered_rules`."""

    triggered: bool
    """Whether this rule's condition matched the given inputs."""

    reason: str
    """Human-readable explanation of the outcome, whether triggered or not."""

    signal: Signal
    """What this rule advocates. Only meaningful when `triggered` is `True`;
    untriggered rules always report `"neutral"`."""

    def as_dict(self) -> dict[str, Any]:
        """Return this result as a plain, JSON-serializable dict."""
        return {
            "rule_name": self.rule_name,
            "triggered": self.triggered,
            "reason": self.reason,
            "signal": self.signal,
        }


def rule_low_aggregated_quality(
    aggregated_quality_score: float | None,
    threshold: float = QUALITY_THRESHOLD,
) -> RuleResult:
    """Rule 1: a low `aggregated_quality_score` requires correction.

    Args:
        aggregated_quality_score: `state.aggregated_quality_score`.
        threshold: The minimum acceptable aggregated quality score.

    Returns:
        Triggered (`signal="correct"`) if `aggregated_quality_score` is
        below `threshold`; not triggered if it is at or above `threshold`,
        or if it is `None` (no score available yet).
    """
    name = "low_aggregated_quality"
    if aggregated_quality_score is None:
        return RuleResult(name, False, "No aggregated quality score is available yet.", "neutral")
    if aggregated_quality_score < threshold:
        return RuleResult(
            name,
            True,
            f"Aggregated quality score {aggregated_quality_score!r} is below "
            f"threshold {threshold!r}.",
            "correct",
        )
    return RuleResult(
        name,
        False,
        f"Aggregated quality score {aggregated_quality_score!r} meets threshold {threshold!r}.",
        "neutral",
    )


def rule_max_iterations_reached(iteration_count: int, max_iterations: int) -> RuleResult:
    """Rule 2: reaching the iteration budget forbids further correction.

    This is a hard stop: when triggered, `CorrectionDecisionEngine` treats
    it as overriding every other rule, since retrying indefinitely is not
    an option regardless of quality signals.

    Args:
        iteration_count: `state.iteration_count`.
        max_iterations: `state.max_iterations`.

    Returns:
        Triggered (`signal="no_correct"`) if `iteration_count >= max_iterations`.
    """
    name = "max_iterations_reached"
    if iteration_count >= max_iterations:
        return RuleResult(
            name,
            True,
            f"Iteration count {iteration_count} has reached max_iterations {max_iterations}.",
            "no_correct",
        )
    return RuleResult(
        name,
        False,
        f"Iteration count {iteration_count} is below max_iterations {max_iterations}.",
        "neutral",
    )


def rule_meta_critic_escalation(
    critic_scores: dict[str, float],
    threshold: float = META_CRITIC_ESCALATION_THRESHOLD,
) -> RuleResult:
    """Rule 3: a high `MetaCritic` score is treated as an escalation signal requiring correction.

    Args:
        critic_scores: `state.critic_scores`.
        threshold: The escalation threshold for `MetaCritic`'s score.

    Returns:
        Triggered (`signal="correct"`) if `critic_scores["MetaCritic"]`
        exceeds `threshold`; not triggered if it is at or below
        `threshold`, or if `MetaCritic` has no recorded score.
    """
    name = "meta_critic_escalation"
    meta_score = critic_scores.get("MetaCritic")
    if meta_score is None:
        return RuleResult(name, False, "No MetaCritic score is available.", "neutral")
    if meta_score > threshold:
        return RuleResult(
            name,
            True,
            f"MetaCritic score {meta_score!r} exceeds threshold {threshold!r}.",
            "correct",
        )
    return RuleResult(
        name,
        False,
        f"MetaCritic score {meta_score!r} does not exceed threshold {threshold!r}.",
        "neutral",
    )


def rule_requires_self_correction(requires_self_correction: bool) -> RuleResult:
    """Rule 4: an error feature explicitly flagging `requires_self_correction` requires correction.

    Args:
        requires_self_correction: The latest error feature's
            `requires_self_correction` flag (see
            `_extract_requires_self_correction` in
            `app/correction_policy/decision.py`).

    Returns:
        Triggered (`signal="correct"`) if `requires_self_correction` is `True`.
    """
    name = "requires_self_correction"
    if requires_self_correction:
        return RuleResult(
            name, True, "The latest error feature flags requires_self_correction.", "correct"
        )
    return RuleResult(name, False, "No error feature flags requires_self_correction.", "neutral")


def rule_low_memory_relevance(
    memory_context: dict[str, Any],
    threshold: float = LOW_MEMORY_RELEVANCE_THRESHOLD,
) -> RuleResult:
    """Extra rule: an explicitly low `memory_relevance` signal requires correction.

    Reads the same forward-compatible `memory_context["memory_relevance"]`
    hook established by `app/policy_engine/scorer.py`. Not one of the five
    example rules in the spec, but included so `memory_context` (one of
    the engine's declared inputs) is genuinely consulted rather than
    accepted-but-unused.

    Args:
        memory_context: `state.memory_context`.
        threshold: The minimum acceptable memory relevance.

    Returns:
        Triggered (`signal="correct"`) if `memory_context["memory_relevance"]`
        is present and below `threshold`.
    """
    name = "low_memory_relevance"
    value = memory_context.get("memory_relevance")
    if not isinstance(value, (int, float)):
        return RuleResult(
            name, False, "No memory_relevance signal is available in memory_context.", "neutral"
        )
    if value < threshold:
        return RuleResult(
            name,
            True,
            f"memory_context memory_relevance {value!r} is below threshold {threshold!r}.",
            "correct",
        )
    return RuleResult(
        name,
        False,
        f"memory_context memory_relevance {value!r} meets threshold {threshold!r}.",
        "neutral",
    )


def rule_all_critics_high_quality(
    critic_scores: dict[str, float],
    threshold: float = QUALITY_THRESHOLD,
) -> RuleResult:
    """Rule 5: if every critic score exceeds the quality threshold, correction is unnecessary.

    Args:
        critic_scores: `state.critic_scores`.
        threshold: The minimum quality score every critic must exceed.

    Returns:
        Triggered (`signal="no_correct"`) if `critic_scores` is non-empty
        and every value exceeds `threshold`; not triggered if it is empty
        or any value is at or below `threshold`.
    """
    name = "all_critics_high_quality"
    if not critic_scores:
        return RuleResult(name, False, "No critic scores are available.", "neutral")
    if all(score > threshold for score in critic_scores.values()):
        return RuleResult(
            name,
            True,
            f"All {len(critic_scores)} critic score(s) exceed threshold {threshold!r}.",
            "no_correct",
        )
    return RuleResult(
        name, False, f"Not every critic score exceeds threshold {threshold!r}.", "neutral"
    )
