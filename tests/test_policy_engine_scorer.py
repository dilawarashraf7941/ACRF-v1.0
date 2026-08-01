"""Unit tests for `HeuristicPolicyScorer` (app/policy_engine/scorer.py)."""

import pytest

from app.policy_engine.scorer import (
    HeuristicPolicyScorer,
    _extract_attempt_pressure,
    _extract_iteration_pressure,
    _extract_plan_complexity,
)
from app.state import AgentState, ErrorFeature, PlannerOutput, WorkerOutput

CANDIDATES = ("LogicCritic", "CodeCritic", "FactCritic", "MetaCritic")


def _make_state(**overrides: object) -> AgentState:
    defaults: dict[str, object] = {"session_id": "s", "task_id": "t", "user_query": "q"}
    defaults.update(overrides)
    return AgentState(**defaults)  # type: ignore[arg-type]


def _error_feature(**metadata_overrides: object) -> ErrorFeature:
    metadata: dict[str, object] = {
        "confidence": 1.0,
        "risk_level": "low",
        "output_type": "text",
        "error_category": "none",
        "profile": {},
    }
    metadata.update(metadata_overrides)
    return ErrorFeature(
        error_type="none",
        description="test fixture",
        severity="low",
        source_node="error_feature_extractor",
        metadata=metadata,
    )


# --- Basic contract ---


def test_score_returns_one_entry_per_candidate() -> None:
    scorer = HeuristicPolicyScorer()

    scores = scorer.score(_make_state(), CANDIDATES)

    assert set(scores.keys()) == set(CANDIDATES)
    assert all(isinstance(v, float) for v in scores.values())


def test_scores_are_bounded_between_zero_and_one() -> None:
    scorer = HeuristicPolicyScorer()
    state = _make_state()
    state.error_features = [
        _error_feature(
            confidence=0.0,
            risk_level="critical",
            profile={
                "task_complexity": "very_complex",
                "requires_meta_critic": True,
                "requires_self_correction": True,
                "memory_relevance": 1.0,
            },
        )
    ]
    state.iteration_count = 100
    state.worker_outputs = [WorkerOutput(worker_id="w") for _ in range(20)]

    scores = scorer.score(state, CANDIDATES)

    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_scores_not_all_zero_for_neutral_state() -> None:
    scorer = HeuristicPolicyScorer()

    scores = scorer.score(_make_state(), CANDIDATES)

    assert any(v != 0.0 for v in scores.values())


def test_unknown_critic_name_gets_a_score_via_default_weights() -> None:
    scorer = HeuristicPolicyScorer()

    scores = scorer.score(_make_state(), ["SomeFutureCritic"])

    assert "SomeFutureCritic" in scores
    assert isinstance(scores["SomeFutureCritic"], float)


def test_empty_candidate_list_returns_empty_scores() -> None:
    scorer = HeuristicPolicyScorer()

    assert scorer.score(_make_state(), []) == {}


# --- Determinism ---


def test_is_deterministic_for_identical_state() -> None:
    scorer = HeuristicPolicyScorer()
    state_a = _make_state()
    state_a.iteration_count = 2
    state_b = _make_state()
    state_b.iteration_count = 2

    assert scorer.score(state_a, CANDIDATES) == scorer.score(state_b, CANDIDATES)


# --- Differentiation: different states -> different scores ---


def test_high_risk_state_differs_from_low_risk_state() -> None:
    scorer = HeuristicPolicyScorer()
    low_risk = _make_state()
    low_risk.error_features = [_error_feature(risk_level="low")]
    high_risk = _make_state()
    high_risk.error_features = [_error_feature(risk_level="critical")]

    assert scorer.score(low_risk, CANDIDATES) != scorer.score(high_risk, CANDIDATES)


def test_code_output_boosts_code_critic_relative_to_others() -> None:
    scorer = HeuristicPolicyScorer()
    code_state = _make_state()
    code_state.error_features = [_error_feature(output_type="code")]
    text_state = _make_state()
    text_state.error_features = [_error_feature(output_type="text")]

    code_scores = scorer.score(code_state, CANDIDATES)
    text_scores = scorer.score(text_state, CANDIDATES)

    assert code_scores["CodeCritic"] > text_scores["CodeCritic"]
    # Relative to LogicCritic within the same state, code output favors CodeCritic more.
    assert (code_scores["CodeCritic"] - code_scores["LogicCritic"]) > (
        text_scores["CodeCritic"] - text_scores["LogicCritic"]
    )


def test_requires_meta_critic_flag_boosts_meta_critic() -> None:
    scorer = HeuristicPolicyScorer()
    baseline = _make_state()
    baseline.error_features = [_error_feature()]
    flagged = _make_state()
    flagged.error_features = [_error_feature(profile={"requires_meta_critic": True})]

    flagged_score = scorer.score(flagged, CANDIDATES)["MetaCritic"]
    baseline_score = scorer.score(baseline, CANDIDATES)["MetaCritic"]
    assert flagged_score > baseline_score


def test_requires_self_correction_flag_boosts_meta_critic() -> None:
    scorer = HeuristicPolicyScorer()
    baseline = _make_state()
    baseline.error_features = [_error_feature()]
    flagged = _make_state()
    flagged.error_features = [_error_feature(profile={"requires_self_correction": True})]

    flagged_score = scorer.score(flagged, CANDIDATES)["MetaCritic"]
    baseline_score = scorer.score(baseline, CANDIDATES)["MetaCritic"]
    assert flagged_score > baseline_score


def test_memory_relevance_from_profile_boosts_fact_critic() -> None:
    scorer = HeuristicPolicyScorer()
    baseline = _make_state()
    baseline.error_features = [_error_feature(profile={"memory_relevance": 0.0})]
    relevant = _make_state()
    relevant.error_features = [_error_feature(profile={"memory_relevance": 0.9})]

    relevant_score = scorer.score(relevant, CANDIDATES)["FactCritic"]
    baseline_score = scorer.score(baseline, CANDIDATES)["FactCritic"]
    assert relevant_score > baseline_score


def test_memory_relevance_from_memory_context_boosts_fact_critic() -> None:
    scorer = HeuristicPolicyScorer()
    baseline = _make_state()
    relevant = _make_state()
    relevant.memory_context = {"memory_relevance": 0.9}

    relevant_score = scorer.score(relevant, CANDIDATES)["FactCritic"]
    baseline_score = scorer.score(baseline, CANDIDATES)["FactCritic"]
    assert relevant_score > baseline_score


def test_task_complexity_from_planner_output_decomposition_increases_scores() -> None:
    scorer = HeuristicPolicyScorer()
    simple = _make_state()
    simple.planner_output = PlannerOutput(decomposition=[])
    complex_ = _make_state()
    complex_.planner_output = PlannerOutput(
        decomposition=["step1", "step2", "step3", "step4", "step5"]
    )

    simple_scores = scorer.score(simple, CANDIDATES)
    complex_scores = scorer.score(complex_, CANDIDATES)

    assert complex_scores["LogicCritic"] > simple_scores["LogicCritic"]
    assert complex_scores["CodeCritic"] > simple_scores["CodeCritic"]


def test_iteration_count_increases_meta_critic_score() -> None:
    scorer = HeuristicPolicyScorer()
    low_iteration = _make_state()
    low_iteration.iteration_count = 0
    high_iteration = _make_state()
    high_iteration.iteration_count = 5

    assert (
        scorer.score(high_iteration, CANDIDATES)["MetaCritic"]
        > scorer.score(low_iteration, CANDIDATES)["MetaCritic"]
    )


def test_extra_worker_attempts_increase_meta_critic_score() -> None:
    scorer = HeuristicPolicyScorer()
    single_attempt = _make_state()
    single_attempt.worker_outputs = [WorkerOutput(worker_id="w")]
    many_attempts = _make_state()
    many_attempts.worker_outputs = [WorkerOutput(worker_id="w") for _ in range(5)]

    assert (
        scorer.score(many_attempts, CANDIDATES)["MetaCritic"]
        > scorer.score(single_attempt, CANDIDATES)["MetaCritic"]
    )


def test_uncertainty_increases_logic_and_fact_critic_scores() -> None:
    scorer = HeuristicPolicyScorer()
    confident = _make_state()
    confident.error_features = [_error_feature(confidence=1.0)]
    uncertain = _make_state()
    uncertain.error_features = [_error_feature(confidence=0.1)]

    confident_scores = scorer.score(confident, CANDIDATES)
    uncertain_scores = scorer.score(uncertain, CANDIDATES)

    assert uncertain_scores["LogicCritic"] > confident_scores["LogicCritic"]
    assert uncertain_scores["FactCritic"] > confident_scores["FactCritic"]


def test_scorer_does_not_read_task_type() -> None:
    """task_type is not in the scorer's declared input set; scores must be
    identical regardless of it.
    """
    scorer = HeuristicPolicyScorer()
    code_task = _make_state(task_type="code")
    other_task = _make_state(task_type="research")

    assert scorer.score(code_task, CANDIDATES) == scorer.score(other_task, CANDIDATES)


# --- Feature extraction edge cases ---


def test_confidence_falls_back_to_profile_confidence_score() -> None:
    scorer = HeuristicPolicyScorer()
    state = _make_state()
    feature = ErrorFeature(
        error_type="none",
        description="d",
        source_node="error_feature_extractor",
        metadata={"profile": {"confidence_score": 0.3}},
    )
    state.error_features = [feature]

    features = scorer.extract_features(state)

    assert features.uncertainty == pytest.approx(0.7)


def test_missing_confidence_defaults_to_zero_uncertainty() -> None:
    scorer = HeuristicPolicyScorer()
    state = _make_state()
    state.error_features = [
        ErrorFeature(error_type="none", description="d", source_node="s", metadata={})
    ]

    assert scorer.extract_features(state).uncertainty == 0.0


def test_unrecognized_risk_level_defaults_to_zero() -> None:
    scorer = HeuristicPolicyScorer()
    state = _make_state()
    state.error_features = [_error_feature(risk_level="not_a_real_level")]

    assert scorer.extract_features(state).risk == 0.0


def test_extract_plan_complexity_caps_at_one() -> None:
    planner_output = PlannerOutput(decomposition=[f"step{i}" for i in range(50)])

    assert _extract_plan_complexity(planner_output) == 1.0


def test_extract_plan_complexity_handles_none() -> None:
    assert _extract_plan_complexity(None) == 0.0


def test_extract_iteration_pressure_caps_at_one() -> None:
    assert _extract_iteration_pressure(1000) == 1.0
    assert _extract_iteration_pressure(0) == 0.0
    assert _extract_iteration_pressure(-1) == 0.0


def test_extract_attempt_pressure_ignores_single_output() -> None:
    assert _extract_attempt_pressure([WorkerOutput(worker_id="w")]) == 0.0
    assert _extract_attempt_pressure([]) == 0.0


def test_extract_attempt_pressure_caps_at_one() -> None:
    outputs = [WorkerOutput(worker_id="w") for _ in range(50)]

    assert _extract_attempt_pressure(outputs) == 1.0
