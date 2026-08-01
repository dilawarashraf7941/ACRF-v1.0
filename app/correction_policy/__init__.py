"""The Correction Decision Policy: deterministic, rule-based replacement for the
previous "always correct" placeholder behavior in `self_correction_node`.

No reinforcement learning, no Q-learning, no PPO/DQN, no neural networks,
no LLM calls — every rule and threshold in this module is a fixed,
hand-authored constant.
"""

from app.correction_policy.decision import CorrectionDecision, CorrectionDecisionEngine
from app.correction_policy.rules import (
    LOW_MEMORY_RELEVANCE_THRESHOLD,
    META_CRITIC_ESCALATION_THRESHOLD,
    QUALITY_THRESHOLD,
    RuleResult,
    rule_all_critics_high_quality,
    rule_low_aggregated_quality,
    rule_low_memory_relevance,
    rule_max_iterations_reached,
    rule_meta_critic_escalation,
    rule_requires_self_correction,
)

__all__ = [
    "LOW_MEMORY_RELEVANCE_THRESHOLD",
    "META_CRITIC_ESCALATION_THRESHOLD",
    "QUALITY_THRESHOLD",
    "CorrectionDecision",
    "CorrectionDecisionEngine",
    "RuleResult",
    "rule_all_critics_high_quality",
    "rule_low_aggregated_quality",
    "rule_low_memory_relevance",
    "rule_max_iterations_reached",
    "rule_meta_critic_escalation",
    "rule_requires_self_correction",
]
