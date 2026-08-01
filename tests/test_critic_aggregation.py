"""Unit tests for the ACRF critic aggregation infrastructure:
`AggregatedCriticResult`, `AggregationStrategy`, and the four placeholder
strategies (`MajorityVoteStrategy`, `WeightedAverageStrategy`,
`MaxScoreStrategy`, `PolicyWeightedStrategy`).
"""

import pytest
from pydantic import ValidationError

from app.critics import (
    AggregatedCriticResult,
    AggregationStrategy,
    CriticResult,
    CriticType,
    MajorityVoteStrategy,
    MaxScoreStrategy,
    PolicyWeightedStrategy,
    WeightedAverageStrategy,
)

ALL_STRATEGY_CLASSES = (
    MajorityVoteStrategy,
    WeightedAverageStrategy,
    MaxScoreStrategy,
    PolicyWeightedStrategy,
)


def _make_results() -> list[CriticResult]:
    return [
        CriticResult(critic_name="LogicCritic", critic_type=CriticType.LOGIC, score=0.9, passed=True),
        CriticResult(critic_name="CodeCritic", critic_type=CriticType.CODE, score=0.2, passed=False),
        CriticResult(critic_name="FactCritic", critic_type=CriticType.FACT, score=0.5, passed=None),
    ]


# --- AggregatedCriticResult ---


def test_aggregated_critic_result_requires_strategy_name() -> None:
    with pytest.raises(ValidationError):
        AggregatedCriticResult()  # type: ignore[call-arg]


def test_aggregated_critic_result_applies_defaults() -> None:
    result = AggregatedCriticResult(strategy_name="SomeStrategy")

    assert result.aggregated_score == 0.0
    assert result.aggregated_passed is None
    assert result.confidence == 0.0
    assert result.contributing_critics == []
    assert result.individual_results == []
    assert result.rationale is None
    assert result.metadata == {}


def test_aggregated_critic_result_accepts_explicit_values() -> None:
    critic_result = CriticResult(critic_name="LogicCritic", score=0.8)

    result = AggregatedCriticResult(
        strategy_name="MajorityVoteStrategy",
        aggregated_score=0.7,
        aggregated_passed=True,
        confidence=0.6,
        contributing_critics=["LogicCritic"],
        individual_results=[critic_result],
        rationale="explanation",
        metadata={"k": "v"},
    )

    assert result.aggregated_score == 0.7
    assert result.aggregated_passed is True
    assert result.confidence == 0.6
    assert result.contributing_critics == ["LogicCritic"]
    assert result.individual_results == [critic_result]
    assert result.rationale == "explanation"
    assert result.metadata == {"k": "v"}


@pytest.mark.parametrize("field", ["aggregated_score", "confidence"])
def test_aggregated_critic_result_rejects_out_of_range_bounded_fields(field: str) -> None:
    with pytest.raises(ValidationError):
        AggregatedCriticResult(strategy_name="SomeStrategy", **{field: 1.5})

    with pytest.raises(ValidationError):
        AggregatedCriticResult(strategy_name="SomeStrategy", **{field: -0.1})


def test_aggregated_critic_result_allows_extra_fields() -> None:
    result = AggregatedCriticResult(strategy_name="SomeStrategy", custom_signal="anomaly")

    assert result.custom_signal == "anomaly"  # type: ignore[attr-defined]


def test_aggregated_critic_result_individual_results_round_trip_via_dict() -> None:
    critic_result = CriticResult(critic_name="LogicCritic", score=0.5)
    result = AggregatedCriticResult(strategy_name="SomeStrategy", individual_results=[critic_result])

    dumped = result.model_dump()

    assert dumped["individual_results"][0]["critic_name"] == "LogicCritic"
    assert dumped["individual_results"][0]["score"] == 0.5


# --- AggregationStrategy (abstract) ---


def test_aggregation_strategy_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        AggregationStrategy()  # type: ignore[abstract]


def test_incomplete_subclass_without_aggregate_cannot_be_instantiated() -> None:
    class IncompleteStrategy(AggregationStrategy):
        strategy_name = "IncompleteStrategy"

    with pytest.raises(TypeError):
        IncompleteStrategy()  # type: ignore[abstract]


# --- Concrete placeholder strategies: shared behavior ---


@pytest.mark.parametrize("strategy_class", ALL_STRATEGY_CLASSES)
def test_strategy_is_instantiable(strategy_class: type[AggregationStrategy]) -> None:
    strategy = strategy_class()

    assert isinstance(strategy, AggregationStrategy)


@pytest.mark.parametrize("strategy_class", ALL_STRATEGY_CLASSES)
def test_strategy_aggregate_returns_valid_aggregated_result(
    strategy_class: type[AggregationStrategy],
) -> None:
    strategy = strategy_class()

    result = strategy.aggregate(_make_results())

    assert isinstance(result, AggregatedCriticResult)
    assert result.strategy_name == strategy_class.strategy_name


@pytest.mark.parametrize("strategy_class", ALL_STRATEGY_CLASSES)
def test_strategy_placeholder_result_is_neutral(strategy_class: type[AggregationStrategy]) -> None:
    strategy = strategy_class()

    result = strategy.aggregate(_make_results())

    assert result.aggregated_score == 0.0
    assert result.aggregated_passed is None
    assert result.confidence == 0.0
    assert "placeholder" in result.rationale.lower()
    assert result.metadata["strategy_class"] == strategy_class.__name__


@pytest.mark.parametrize("strategy_class", ALL_STRATEGY_CLASSES)
def test_strategy_does_not_compute_from_individual_scores(
    strategy_class: type[AggregationStrategy],
) -> None:
    """No real aggregation algorithm: wildly different input scores/pass
    values must not change the fixed placeholder score/passed/confidence.
    """
    strategy = strategy_class()

    all_high = [
        CriticResult(critic_name="A", score=1.0, passed=True, confidence=1.0),
        CriticResult(critic_name="B", score=1.0, passed=True, confidence=1.0),
    ]
    all_low = [
        CriticResult(critic_name="C", score=0.0, passed=False, confidence=0.0),
        CriticResult(critic_name="D", score=0.0, passed=False, confidence=0.0),
    ]

    result_high = strategy.aggregate(all_high)
    result_low = strategy.aggregate(all_low)

    assert result_high.aggregated_score == result_low.aggregated_score == 0.0
    assert result_high.aggregated_passed == result_low.aggregated_passed is None
    assert result_high.confidence == result_low.confidence == 0.0


@pytest.mark.parametrize("strategy_class", ALL_STRATEGY_CLASSES)
def test_strategy_echoes_back_contributing_critics(strategy_class: type[AggregationStrategy]) -> None:
    strategy = strategy_class()
    results = _make_results()

    aggregated = strategy.aggregate(results)

    assert aggregated.contributing_critics == ["LogicCritic", "CodeCritic", "FactCritic"]
    assert aggregated.individual_results == results
    assert aggregated.metadata["result_count"] == 3


@pytest.mark.parametrize("strategy_class", ALL_STRATEGY_CLASSES)
def test_strategy_handles_empty_results_list(strategy_class: type[AggregationStrategy]) -> None:
    strategy = strategy_class()

    result = strategy.aggregate([])

    assert result.contributing_critics == []
    assert result.individual_results == []
    assert result.aggregated_score == 0.0
    assert result.metadata["result_count"] == 0


@pytest.mark.parametrize("strategy_class", ALL_STRATEGY_CLASSES)
def test_strategy_aggregate_is_deterministic_across_instances(
    strategy_class: type[AggregationStrategy],
) -> None:
    results = _make_results()

    result_a = strategy_class().aggregate(results)
    result_b = strategy_class().aggregate(results)

    assert result_a == result_b


# --- Per-strategy identity ---


def test_majority_vote_strategy_identity() -> None:
    assert MajorityVoteStrategy().strategy_name == "MajorityVoteStrategy"


def test_weighted_average_strategy_identity() -> None:
    assert WeightedAverageStrategy().strategy_name == "WeightedAverageStrategy"


def test_max_score_strategy_identity() -> None:
    assert MaxScoreStrategy().strategy_name == "MaxScoreStrategy"


def test_policy_weighted_strategy_identity() -> None:
    assert PolicyWeightedStrategy().strategy_name == "PolicyWeightedStrategy"


def test_strategy_names_are_all_distinct() -> None:
    names = {cls.strategy_name for cls in ALL_STRATEGY_CLASSES}

    assert len(names) == len(ALL_STRATEGY_CLASSES)
