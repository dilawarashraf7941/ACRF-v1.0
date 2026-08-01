"""Unit tests for the ACRF critic infrastructure: `CriticResult`,
`BaseCritic`, and the four placeholder critics (`LogicCritic`,
`CodeCritic`, `FactCritic`, `MetaCritic`).
"""

import pytest
from pydantic import ValidationError

from app.critics import (
    BaseCritic,
    CodeCritic,
    CriticResult,
    CriticType,
    FactCritic,
    LogicCritic,
    MetaCritic,
)

ALL_CRITIC_CLASSES = (LogicCritic, CodeCritic, FactCritic, MetaCritic)


# --- CriticResult ---


def test_critic_result_requires_critic_name() -> None:
    with pytest.raises(ValidationError):
        CriticResult()  # type: ignore[call-arg]


def test_critic_result_applies_defaults() -> None:
    result = CriticResult(critic_name="SomeCritic")

    assert result.critic_type == CriticType.CUSTOM
    assert result.score == 0.0
    assert result.passed is None
    assert result.confidence == 0.0
    assert result.feedback is None
    assert result.metadata == {}


def test_critic_result_accepts_explicit_values() -> None:
    result = CriticResult(
        critic_name="LogicCritic",
        critic_type=CriticType.LOGIC,
        score=0.9,
        passed=True,
        confidence=0.8,
        feedback="Looks sound.",
        metadata={"detail": "value"},
    )

    assert result.critic_name == "LogicCritic"
    assert result.critic_type == CriticType.LOGIC
    assert result.score == 0.9
    assert result.passed is True
    assert result.confidence == 0.8
    assert result.feedback == "Looks sound."
    assert result.metadata == {"detail": "value"}


@pytest.mark.parametrize("field", ["score", "confidence"])
def test_critic_result_rejects_out_of_range_bounded_fields(field: str) -> None:
    with pytest.raises(ValidationError):
        CriticResult(critic_name="SomeCritic", **{field: 1.5})

    with pytest.raises(ValidationError):
        CriticResult(critic_name="SomeCritic", **{field: -0.1})


def test_critic_result_rejects_invalid_critic_type() -> None:
    with pytest.raises(ValidationError):
        CriticResult(critic_name="SomeCritic", critic_type="not_a_real_type")


def test_critic_result_allows_extra_fields() -> None:
    result = CriticResult(critic_name="SomeCritic", custom_signal="anomaly")

    assert result.custom_signal == "anomaly"  # type: ignore[attr-defined]


# --- BaseCritic (abstract) ---


def test_base_critic_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        BaseCritic()  # type: ignore[abstract]


def test_incomplete_subclass_without_evaluate_cannot_be_instantiated() -> None:
    class IncompleteCritic(BaseCritic):
        critic_name = "IncompleteCritic"

    with pytest.raises(TypeError):
        IncompleteCritic()  # type: ignore[abstract]


# --- Concrete placeholder critics: shared behavior ---


@pytest.mark.parametrize("critic_class", ALL_CRITIC_CLASSES)
def test_critic_is_instantiable(critic_class: type[BaseCritic]) -> None:
    critic = critic_class()

    assert isinstance(critic, BaseCritic)


@pytest.mark.parametrize("critic_class", ALL_CRITIC_CLASSES)
def test_critic_evaluate_returns_valid_critic_result(critic_class: type[BaseCritic]) -> None:
    critic = critic_class()

    result = critic.evaluate("some content to evaluate")

    assert isinstance(result, CriticResult)
    assert result.critic_name == critic_class.critic_name
    assert result.critic_type == critic_class.critic_type


@pytest.mark.parametrize("critic_class", ALL_CRITIC_CLASSES)
def test_critic_placeholder_result_is_neutral(critic_class: type[BaseCritic]) -> None:
    critic = critic_class()

    result = critic.evaluate("anything")

    assert result.score == 0.0
    assert result.passed is None
    assert result.confidence == 0.0
    assert "placeholder" in result.feedback.lower()
    assert result.metadata["critic_class"] == critic_class.__name__


@pytest.mark.parametrize("critic_class", ALL_CRITIC_CLASSES)
@pytest.mark.parametrize(
    "content",
    [
        "plain text",
        "",
        None,
        {"nested": {"data": [1, 2, 3]}},
        12345,
        ["a", "list", "of", "things"],
    ],
)
def test_critic_evaluate_ignores_content(critic_class: type[BaseCritic], content: object) -> None:
    critic = critic_class()

    result = critic.evaluate(content)

    assert result == critic.evaluate("a completely different value")


@pytest.mark.parametrize("critic_class", ALL_CRITIC_CLASSES)
def test_critic_evaluate_is_deterministic_across_instances(critic_class: type[BaseCritic]) -> None:
    result_a = critic_class().evaluate("x")
    result_b = critic_class().evaluate("y")

    assert result_a == result_b


# --- Per-critic identity ---


def test_logic_critic_identity() -> None:
    critic = LogicCritic()

    assert critic.critic_name == "LogicCritic"
    assert critic.critic_type == CriticType.LOGIC


def test_code_critic_identity() -> None:
    critic = CodeCritic()

    assert critic.critic_name == "CodeCritic"
    assert critic.critic_type == CriticType.CODE


def test_fact_critic_identity() -> None:
    critic = FactCritic()

    assert critic.critic_name == "FactCritic"
    assert critic.critic_type == CriticType.FACT


def test_meta_critic_identity() -> None:
    critic = MetaCritic()

    assert critic.critic_name == "MetaCritic"
    assert critic.critic_type == CriticType.META


def test_critic_names_are_all_distinct() -> None:
    names = {cls.critic_name for cls in ALL_CRITIC_CLASSES}

    assert len(names) == len(ALL_CRITIC_CLASSES)


def test_critic_types_are_all_distinct() -> None:
    types_ = {cls.critic_type for cls in ALL_CRITIC_CLASSES}

    assert len(types_) == len(ALL_CRITIC_CLASSES)
