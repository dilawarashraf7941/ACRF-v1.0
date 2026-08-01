"""Unit tests for the individual correction-policy rules (app/correction_policy/rules.py).

Each rule is exercised directly, in isolation, against primitive inputs —
no `AgentState` construction required.
"""

from app.correction_policy.rules import (
    LOW_MEMORY_RELEVANCE_THRESHOLD,
    META_CRITIC_ESCALATION_THRESHOLD,
    QUALITY_THRESHOLD,
    rule_all_critics_high_quality,
    rule_low_aggregated_quality,
    rule_low_memory_relevance,
    rule_max_iterations_reached,
    rule_meta_critic_escalation,
    rule_requires_self_correction,
)

# --- rule_low_aggregated_quality ---


def test_low_quality_below_threshold_triggers() -> None:
    result = rule_low_aggregated_quality(0.3)

    assert result.triggered is True
    assert result.signal == "correct"


def test_low_quality_at_threshold_does_not_trigger() -> None:
    result = rule_low_aggregated_quality(QUALITY_THRESHOLD)

    assert result.triggered is False
    assert result.signal == "neutral"


def test_low_quality_above_threshold_does_not_trigger() -> None:
    result = rule_low_aggregated_quality(0.95)

    assert result.triggered is False


def test_low_quality_none_does_not_trigger() -> None:
    result = rule_low_aggregated_quality(None)

    assert result.triggered is False
    assert result.signal == "neutral"


def test_low_quality_respects_custom_threshold() -> None:
    assert rule_low_aggregated_quality(0.5, threshold=0.4).triggered is False
    assert rule_low_aggregated_quality(0.3, threshold=0.4).triggered is True


# --- rule_max_iterations_reached ---


def test_max_iterations_reached_triggers_when_equal() -> None:
    result = rule_max_iterations_reached(5, 5)

    assert result.triggered is True
    assert result.signal == "no_correct"


def test_max_iterations_reached_triggers_when_exceeded() -> None:
    assert rule_max_iterations_reached(6, 5).triggered is True


def test_max_iterations_not_reached_below_limit() -> None:
    result = rule_max_iterations_reached(2, 5)

    assert result.triggered is False
    assert result.signal == "neutral"


def test_max_iterations_zero_max_triggers_immediately() -> None:
    assert rule_max_iterations_reached(0, 0).triggered is True


# --- rule_meta_critic_escalation ---


def test_meta_critic_escalation_triggers_above_threshold() -> None:
    result = rule_meta_critic_escalation({"MetaCritic": 0.9})

    assert result.triggered is True
    assert result.signal == "correct"


def test_meta_critic_escalation_does_not_trigger_at_threshold() -> None:
    result = rule_meta_critic_escalation({"MetaCritic": META_CRITIC_ESCALATION_THRESHOLD})

    assert result.triggered is False


def test_meta_critic_escalation_does_not_trigger_below_threshold() -> None:
    assert rule_meta_critic_escalation({"MetaCritic": 0.1}).triggered is False


def test_meta_critic_escalation_missing_score_does_not_trigger() -> None:
    result = rule_meta_critic_escalation({"LogicCritic": 0.99})

    assert result.triggered is False
    assert result.signal == "neutral"


def test_meta_critic_escalation_empty_scores_does_not_trigger() -> None:
    assert rule_meta_critic_escalation({}).triggered is False


# --- rule_requires_self_correction ---


def test_requires_self_correction_true_triggers() -> None:
    result = rule_requires_self_correction(True)

    assert result.triggered is True
    assert result.signal == "correct"


def test_requires_self_correction_false_does_not_trigger() -> None:
    result = rule_requires_self_correction(False)

    assert result.triggered is False
    assert result.signal == "neutral"


# --- rule_low_memory_relevance ---


def test_low_memory_relevance_triggers_below_threshold() -> None:
    result = rule_low_memory_relevance({"memory_relevance": 0.05})

    assert result.triggered is True
    assert result.signal == "correct"


def test_low_memory_relevance_does_not_trigger_at_threshold() -> None:
    result = rule_low_memory_relevance({"memory_relevance": LOW_MEMORY_RELEVANCE_THRESHOLD})

    assert result.triggered is False


def test_low_memory_relevance_does_not_trigger_above_threshold() -> None:
    assert rule_low_memory_relevance({"memory_relevance": 0.8}).triggered is False


def test_low_memory_relevance_missing_signal_does_not_trigger() -> None:
    result = rule_low_memory_relevance({})

    assert result.triggered is False
    assert result.signal == "neutral"


def test_low_memory_relevance_ignores_non_numeric_value() -> None:
    result = rule_low_memory_relevance({"memory_relevance": "not_a_number"})

    assert result.triggered is False


# --- rule_all_critics_high_quality ---


def test_all_critics_high_quality_triggers_when_all_exceed_threshold() -> None:
    result = rule_all_critics_high_quality({"LogicCritic": 0.9, "CodeCritic": 0.8})

    assert result.triggered is True
    assert result.signal == "no_correct"


def test_all_critics_high_quality_does_not_trigger_if_one_is_low() -> None:
    result = rule_all_critics_high_quality({"LogicCritic": 0.9, "CodeCritic": 0.1})

    assert result.triggered is False


def test_all_critics_high_quality_does_not_trigger_at_threshold() -> None:
    result = rule_all_critics_high_quality({"LogicCritic": QUALITY_THRESHOLD})

    assert result.triggered is False


def test_all_critics_high_quality_empty_scores_does_not_trigger() -> None:
    result = rule_all_critics_high_quality({})

    assert result.triggered is False
    assert result.signal == "neutral"


# --- RuleResult.as_dict ---


def test_rule_result_as_dict_matches_fields() -> None:
    result = rule_low_aggregated_quality(0.1)

    as_dict = result.as_dict()

    assert as_dict == {
        "rule_name": "low_aggregated_quality",
        "triggered": True,
        "reason": as_dict["reason"],
        "signal": "correct",
    }
