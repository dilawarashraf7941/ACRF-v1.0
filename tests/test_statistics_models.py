"""Unit tests for `StatisticalComparison` (`app/evaluation/statistics/models.py`)."""

import pytest
from pydantic import ValidationError

from app.evaluation.experiments import ConfidenceInterval
from app.evaluation.statistics.models import StatisticalComparison


def _make_comparison(**overrides: object) -> StatisticalComparison:
    defaults: dict[str, object] = {
        "baseline_policy": "HeuristicPolicy",
        "candidate_policy": "LinUCBPolicy",
        "sample_size": 10,
        "mean_difference": 0.05,
        "confidence_interval": ConfidenceInterval(lower=0.01, upper=0.09),
        "p_value": 0.03,
        "effect_size": 0.5,
        "test_used": "paired_t_test",
        "significant": True,
    }
    defaults.update(overrides)
    return StatisticalComparison(**defaults)


def test_constructs_with_required_fields() -> None:
    comparison = _make_comparison()
    assert comparison.baseline_policy == "HeuristicPolicy"
    assert comparison.metadata == {}


def test_missing_required_field_raises() -> None:
    with pytest.raises(ValidationError):
        StatisticalComparison(baseline_policy="A", candidate_policy="B")


def test_negative_sample_size_raises() -> None:
    with pytest.raises(ValidationError):
        _make_comparison(sample_size=-1)


@pytest.mark.parametrize("p_value", [-0.01, 1.01])
def test_p_value_out_of_bounds_raises(p_value: float) -> None:
    with pytest.raises(ValidationError):
        _make_comparison(p_value=p_value)


@pytest.mark.parametrize("p_value", [0.0, 0.5, 1.0])
def test_p_value_boundary_values_are_valid(p_value: float) -> None:
    comparison = _make_comparison(p_value=p_value)
    assert comparison.p_value == p_value


def test_is_frozen() -> None:
    comparison = _make_comparison()
    with pytest.raises(ValidationError):
        comparison.significant = False  # type: ignore[misc]


def test_holds_confidence_interval() -> None:
    interval = ConfidenceInterval(lower=-0.1, upper=0.2, confidence_level=0.95)
    comparison = _make_comparison(confidence_interval=interval)
    assert comparison.confidence_interval == interval


def test_allows_extra_fields() -> None:
    comparison = StatisticalComparison(
        baseline_policy="A",
        candidate_policy="B",
        sample_size=1,
        mean_difference=0.0,
        confidence_interval=ConfidenceInterval(lower=0.0, upper=0.0),
        p_value=1.0,
        effect_size=0.0,
        test_used="insufficient_data",
        significant=False,
        extra_signal="anything",  # type: ignore[call-arg]
    )
    assert comparison.model_dump()["extra_signal"] == "anything"
