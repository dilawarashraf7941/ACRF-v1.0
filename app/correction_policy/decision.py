"""The Correction Decision Policy: `CorrectionDecision` and `CorrectionDecisionEngine`.

Replaces the previous "always correct" placeholder behavior in
`self_correction_node` with a deterministic, rule-based decision (see
`app/correction_policy/rules.py`). No reinforcement learning, no
Q-learning, no PPO/DQN, no neural networks, no LLM calls, no randomness —
every rule and threshold is a fixed constant, so identical `AgentState`
values always produce identical decisions.

Only six `AgentState` fields are read: `aggregated_quality_score`,
`critic_scores`, `iteration_count`, `max_iterations`, `memory_context`,
and `error_features`.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.correction_policy.rules import (
    RuleResult,
    rule_all_critics_high_quality,
    rule_low_aggregated_quality,
    rule_low_memory_relevance,
    rule_max_iterations_reached,
    rule_meta_critic_escalation,
    rule_requires_self_correction,
)
from app.state import AgentState, ErrorFeature


class CorrectionDecision(BaseModel):
    """The outcome of `CorrectionDecisionEngine.decide`."""

    model_config = ConfigDict(extra="allow")

    should_correct: bool = Field(
        ...,
        description="Whether a correction should be applied.",
    )
    reason: str = Field(
        ...,
        description="Human-readable explanation of the decision.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence, from 0.0 to 1.0, in this decision.",
    )
    decision_strategy: str = Field(
        ...,
        description=(
            "Identifier of which branch of the decision policy produced this outcome "
            "(e.g. 'hard_stop_max_iterations', 'rule_based_correction', "
            "'rule_based_finish', 'default_no_signal')."
        ),
    )
    triggered_rules: list[str] = Field(
        default_factory=list,
        description=(
            "Names of every rule whose condition matched, regardless of "
            "which one determined the final decision."
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional diagnostic data, including the full per-rule results.",
    )


def _latest_error_feature(error_features: list[ErrorFeature]) -> ErrorFeature | None:
    """Return the most recently extracted `ErrorFeature`, or `None` if there is none."""
    return error_features[-1] if error_features else None


def _extract_requires_self_correction(error_features: list[ErrorFeature]) -> bool:
    """Read `requires_self_correction` from the latest error feature's nested profile.

    Mirrors `_extract_requires_flag` in `app/policy_engine/scorer.py`; kept
    as an independent implementation here so this module has no
    dependency on `app/policy_engine`.

    Args:
        error_features: `state.error_features`.

    Returns:
        `True` if the latest error feature's `metadata["profile"]["requires_self_correction"]`
        is truthy, else `False`.
    """
    latest = _latest_error_feature(error_features)
    if latest is None:
        return False
    profile = latest.metadata.get("profile")
    if not isinstance(profile, dict):
        return False
    return bool(profile.get("requires_self_correction"))


class CorrectionDecisionEngine:
    """Deterministic, rule-based engine deciding whether self-correction should run.

    Reads only `state.aggregated_quality_score`, `state.critic_scores`,
    `state.iteration_count`, `state.max_iterations`,
    `state.memory_context`, and `state.error_features`. Every rule in
    `app/correction_policy/rules.py` is evaluated unconditionally (they
    are cheap, pure functions), and the results are combined by a fixed
    priority order:

    1. If `rule_max_iterations_reached` triggers, it is a hard stop:
       `should_correct=False` regardless of every other signal.
    2. Otherwise, if any "correction required" rule triggers
       (`rule_low_aggregated_quality`, `rule_meta_critic_escalation`,
       `rule_requires_self_correction`, `rule_low_memory_relevance`),
       `should_correct=True`.
    3. Otherwise, if `rule_all_critics_high_quality` triggers,
       `should_correct=False` ("finish").
    4. Otherwise, `should_correct=False` by default — no positive signal
       for or against correction was found.

    `triggered_rules` on the returned `CorrectionDecision` always lists
    every rule that matched, even ones outranked by a higher-priority
    rule, for full diagnostic traceability.
    """

    def decide(self, state: AgentState) -> CorrectionDecision:
        """Evaluate the correction decision policy against `state`.

        Args:
            state: The current agent state.

        Returns:
            The resulting `CorrectionDecision`.
        """
        critic_scores = dict(state.critic_scores)
        requires_self_correction = _extract_requires_self_correction(state.error_features)

        max_iterations_result = rule_max_iterations_reached(
            state.iteration_count, state.max_iterations
        )
        correction_rule_results: list[RuleResult] = [
            rule_low_aggregated_quality(state.aggregated_quality_score),
            rule_meta_critic_escalation(critic_scores),
            rule_requires_self_correction(requires_self_correction),
            rule_low_memory_relevance(state.memory_context),
        ]
        finish_result = rule_all_critics_high_quality(critic_scores)

        all_results: list[RuleResult] = [
            max_iterations_result,
            *correction_rule_results,
            finish_result,
        ]
        triggered_rules = [result.rule_name for result in all_results if result.triggered]
        rule_results_metadata = {"rule_results": [result.as_dict() for result in all_results]}

        if max_iterations_result.triggered:
            return CorrectionDecision(
                should_correct=False,
                reason=max_iterations_result.reason,
                confidence=1.0,
                decision_strategy="hard_stop_max_iterations",
                triggered_rules=triggered_rules,
                metadata=rule_results_metadata,
            )

        triggered_correction_rules = [
            result for result in correction_rule_results if result.triggered
        ]
        if triggered_correction_rules:
            confidence = min(1.0, 0.5 + 0.25 * len(triggered_correction_rules))
            reason = " ".join(result.reason for result in triggered_correction_rules)
            return CorrectionDecision(
                should_correct=True,
                reason=reason,
                confidence=confidence,
                decision_strategy="rule_based_correction",
                triggered_rules=triggered_rules,
                metadata={
                    **rule_results_metadata,
                    "correcting_rules": [result.rule_name for result in triggered_correction_rules],
                },
            )

        if finish_result.triggered:
            return CorrectionDecision(
                should_correct=False,
                reason=finish_result.reason,
                confidence=0.9,
                decision_strategy="rule_based_finish",
                triggered_rules=triggered_rules,
                metadata=rule_results_metadata,
            )

        return CorrectionDecision(
            should_correct=False,
            reason="No correction-triggering signal was detected; defaulting to no correction.",
            confidence=0.0,
            decision_strategy="default_no_signal",
            triggered_rules=triggered_rules,
            metadata=rule_results_metadata,
        )
