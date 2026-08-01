"""Unit tests for `AblationConfig`/`AblationResult` (`app/evaluation/ablation/models.py`)."""

import pytest
from pydantic import ValidationError

from app.evaluation.ablation.models import AblationConfig, AblationResult


def _make_config(**overrides: object) -> AblationConfig:
    defaults: dict[str, object] = {
        "experiment_name": "study",
        "baseline_policy": "HeuristicPolicy",
        "candidate_policy": "LinUCBPolicy",
        "ablation_type": "linucb_only",
    }
    defaults.update(overrides)
    return AblationConfig(**defaults)


def _make_result(**overrides: object) -> AblationResult:
    defaults: dict[str, object] = {
        "ablation_type": "linucb_only",
        "baseline_reward": 0.5,
        "candidate_reward": 0.6,
        "reward_difference": 0.1,
        "quality_difference": 0.05,
        "latency_difference": -0.2,
        "iteration_difference": -1.0,
        "conclusion": "LinUCBPolicy performed significantly better.",
    }
    defaults.update(overrides)
    return AblationResult(**defaults)


# --- AblationConfig ---


def test_config_constructs_with_required_fields() -> None:
    config = _make_config()
    assert config.metadata == {}


def test_config_missing_required_field_raises() -> None:
    with pytest.raises(ValidationError):
        AblationConfig(baseline_policy="A", candidate_policy="B", ablation_type="linucb_only")


def test_config_is_frozen() -> None:
    config = _make_config()
    with pytest.raises(ValidationError):
        config.ablation_type = "no_exploration"  # type: ignore[misc]


def test_config_holds_metadata() -> None:
    config = _make_config(metadata={"alpha": 0.5})
    assert config.metadata == {"alpha": 0.5}


# --- AblationResult ---


def test_result_constructs_with_required_fields() -> None:
    result = _make_result()
    assert result.metadata == {}


def test_result_missing_required_field_raises() -> None:
    with pytest.raises(ValidationError):
        AblationResult(ablation_type="linucb_only", baseline_reward=0.5)


def test_result_is_frozen() -> None:
    result = _make_result()
    with pytest.raises(ValidationError):
        result.reward_difference = 1.0  # type: ignore[misc]


def test_result_allows_extra_fields() -> None:
    result = AblationResult(
        ablation_type="linucb_only",
        baseline_reward=0.5,
        candidate_reward=0.6,
        reward_difference=0.1,
        quality_difference=0.0,
        latency_difference=0.0,
        iteration_difference=0.0,
        conclusion="n/a",
        extra_signal="anything",  # type: ignore[call-arg]
    )
    assert result.model_dump()["extra_signal"] == "anything"
