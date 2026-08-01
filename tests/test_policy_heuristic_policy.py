"""Unit tests for `HeuristicPolicy` (`app/policy/heuristic_policy.py`).

The most important test in this file is
`test_matches_pre_refactor_scorer_output_exactly`: it reconstructs the
pre-refactor computation directly (`HeuristicPolicyScorer.score` +
`CriticRanking` + `CriticSelector`, exactly as `policy_engine_node` used
to call them) and asserts `HeuristicPolicy.select_action` — now driven by
a `ContextVector` instead of a live `AgentState` — produces bit-identical
scores, ranking, and selection.
"""

from app.context import ContextEncoder
from app.policy.base import BasePolicy
from app.policy.heuristic_policy import HeuristicPolicy
from app.policy.models import PolicyDecision
from app.policy_engine.ranking import CriticRanking
from app.policy_engine.scorer import HeuristicPolicyScorer
from app.policy_engine.selector import CriticSelector, SelectionStrategy
from app.state import AgentState, ErrorFeature, WorkerOutput

_CANDIDATES = ["LogicCritic", "CodeCritic", "FactCritic", "MetaCritic"]


def _make_state(user_query: str = "q", task_type: str | None = None) -> AgentState:
    return AgentState(
        session_id="session-1", task_id="task-1", user_query=user_query, task_type=task_type
    )


def _high_signal_error_feature() -> ErrorFeature:
    return ErrorFeature(
        error_type="short_output",
        description="test fixture",
        severity="high",
        source_node="error_feature_extractor",
        metadata={
            "confidence": 0.2,
            "risk_level": "high",
            "output_type": "code",
            "error_category": "short_output",
            "profile": {
                "task_complexity": "complex",
                "requires_meta_critic": True,
                "requires_self_correction": True,
                "memory_relevance": 0.9,
                "confidence_score": 0.2,
            },
        },
    )


def _loaded_state() -> AgentState:
    state = _make_state()
    state.error_features = [_high_signal_error_feature()]
    state.iteration_count = 4
    state.worker_outputs = [
        WorkerOutput(worker_id="w"),
        WorkerOutput(worker_id="w"),
        WorkerOutput(worker_id="w"),
    ]
    return state


def _pre_refactor_decision(state: AgentState, candidates: list[str]) -> tuple[dict, list, list]:
    """Reproduce the exact pre-refactor computation `policy_engine_node` used."""
    scores = HeuristicPolicyScorer().score(state, candidates)
    ranking = CriticRanking(scores)
    selected = CriticSelector().select(ranking, SelectionStrategy.TOP_1)
    return scores, ranking.as_list_of_dicts(), selected


def test_is_a_base_policy() -> None:
    assert isinstance(HeuristicPolicy(), BasePolicy)


def test_has_expected_name_and_version() -> None:
    policy = HeuristicPolicy()
    assert policy.policy_name == "HeuristicPolicy"
    assert policy.policy_version == "1.0.0"


def test_returns_policy_decision() -> None:
    context = ContextEncoder().encode(_make_state())

    decision = HeuristicPolicy().select_action(context, _CANDIDATES)

    assert isinstance(decision, PolicyDecision)


def test_matches_pre_refactor_scorer_output_exactly_for_neutral_state() -> None:
    state = _make_state()
    context = ContextEncoder().encode(state)

    decision = HeuristicPolicy().select_action(context, _CANDIDATES)
    expected_scores, expected_ranking, expected_selected = _pre_refactor_decision(
        state, _CANDIDATES
    )

    assert decision.scores == expected_scores
    assert decision.ranking == expected_ranking
    assert decision.selected_critics == expected_selected


def test_matches_pre_refactor_scorer_output_exactly_for_loaded_state() -> None:
    state = _loaded_state()
    context = ContextEncoder().encode(state)

    decision = HeuristicPolicy().select_action(context, _CANDIDATES)
    expected_scores, expected_ranking, expected_selected = _pre_refactor_decision(
        state, _CANDIDATES
    )

    assert decision.scores == expected_scores
    assert decision.ranking == expected_ranking
    assert decision.selected_critics == expected_selected


def test_scores_every_candidate() -> None:
    context = ContextEncoder().encode(_make_state())

    decision = HeuristicPolicy().select_action(context, _CANDIDATES)

    assert set(decision.scores.keys()) == set(_CANDIDATES)


def test_empty_candidate_list_yields_no_selection_and_zero_confidence() -> None:
    context = ContextEncoder().encode(_make_state())

    decision = HeuristicPolicy().select_action(context, [])

    assert decision.selected_critics == []
    assert decision.scores == {}
    assert decision.confidence == 0.0


def test_confidence_equals_top_ranked_score() -> None:
    context = ContextEncoder().encode(_loaded_state())

    decision = HeuristicPolicy().select_action(context, _CANDIDATES)

    top_score = max(decision.scores.values())
    assert decision.confidence == top_score


def test_metadata_records_selection_strategy_and_context_id() -> None:
    context = ContextEncoder().encode(_make_state())

    decision = HeuristicPolicy().select_action(context, _CANDIDATES)

    assert decision.metadata["selection_strategy"] == "top_1"
    assert decision.metadata["context_id"] == context.context_id


def test_is_deterministic() -> None:
    context = ContextEncoder().encode(_loaded_state())

    decision_a = HeuristicPolicy().select_action(context, _CANDIDATES)
    decision_b = HeuristicPolicy().select_action(context, _CANDIDATES)

    assert decision_a.scores == decision_b.scores
    assert decision_a.ranking == decision_b.ranking
    assert decision_a.selected_critics == decision_b.selected_critics


def test_code_signal_boosts_code_critic_above_logic_critic() -> None:
    state = _make_state()
    state.error_features = [_high_signal_error_feature()]  # output_type="code"
    context = ContextEncoder().encode(state)

    decision = HeuristicPolicy().select_action(context, _CANDIDATES)

    assert decision.scores["CodeCritic"] > decision.scores["LogicCritic"]


def test_custom_selection_strategy_is_reflected_in_metadata() -> None:
    context = ContextEncoder().encode(_loaded_state())
    policy = HeuristicPolicy(selection_strategy=SelectionStrategy.TOP_1)

    decision = policy.select_action(context, _CANDIDATES)

    assert decision.metadata["selection_strategy"] == "top_1"


def test_custom_scorer_is_used() -> None:
    context = ContextEncoder().encode(_make_state())
    custom_scorer = HeuristicPolicyScorer()
    policy = HeuristicPolicy(scorer=custom_scorer)

    decision = policy.select_action(context, _CANDIDATES)

    assert set(decision.scores.keys()) == set(_CANDIDATES)
