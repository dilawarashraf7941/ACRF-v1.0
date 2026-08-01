"""Unit tests for the deterministic, heuristic-based `policy_engine_node`
(Algorithm 1, steps 4-6: build candidate critics, score them via
`HeuristicPolicyScorer`, rank and select via `app/policy_engine`).
"""

from app.graph.nodes import _CANDIDATE_CRITIC_NAMES, policy_engine_node
from app.state import AgentState, ErrorFeature, WorkerOutput


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


def test_records_all_candidate_critics() -> None:
    state = _make_state()

    result = policy_engine_node(state)

    candidate_critics = result.memory_context["policy_engine"]["candidate_critics"]
    assert candidate_critics == list(_CANDIDATE_CRITIC_NAMES)


def test_scores_every_candidate() -> None:
    state = _make_state()

    result = policy_engine_node(state)

    scores = result.memory_context["policy_engine"]["scores"]
    assert set(scores.keys()) == set(_CANDIDATE_CRITIC_NAMES)
    assert all(isinstance(value, float) for value in scores.values())


def test_scores_are_not_all_zero() -> None:
    state = _make_state()

    result = policy_engine_node(state)

    scores = result.memory_context["policy_engine"]["scores"]
    assert any(value != 0.0 for value in scores.values())


def test_richer_state_produces_different_scores_than_neutral_state() -> None:
    neutral = policy_engine_node(_make_state())

    loaded_state = _make_state()
    loaded_state.error_features = [_high_signal_error_feature()]
    loaded_state.iteration_count = 4
    loaded_state.worker_outputs = [
        WorkerOutput(worker_id="w"),
        WorkerOutput(worker_id="w"),
        WorkerOutput(worker_id="w"),
    ]
    loaded = policy_engine_node(loaded_state)

    neutral_scores = neutral.memory_context["policy_engine"]["scores"]
    loaded_scores = loaded.memory_context["policy_engine"]["scores"]
    assert neutral_scores != loaded_scores


def test_code_signal_boosts_code_critic_above_neutral() -> None:
    state = _make_state()
    state.error_features = [_high_signal_error_feature()]  # output_type="code"

    result = policy_engine_node(state)

    scores = result.memory_context["policy_engine"]["scores"]
    assert scores["CodeCritic"] > scores["LogicCritic"]


def test_meta_critic_signal_boosts_meta_critic() -> None:
    state = _make_state()
    state.error_features = [_high_signal_error_feature()]  # requires_meta_critic=True

    result = policy_engine_node(state)

    scores = result.memory_context["policy_engine"]["scores"]
    neutral_scores = policy_engine_node(_make_state()).memory_context["policy_engine"]["scores"]
    assert scores["MetaCritic"] > neutral_scores["MetaCritic"]


def test_records_ranking_sorted_highest_first() -> None:
    state = _make_state()
    state.error_features = [_high_signal_error_feature()]

    result = policy_engine_node(state)

    ranking = result.memory_context["policy_engine"]["ranking"]
    scores_in_rank_order = [entry["score"] for entry in ranking]
    assert scores_in_rank_order == sorted(scores_in_rank_order, reverse=True)
    assert [entry["rank"] for entry in ranking] == [1, 2, 3, 4]


def test_records_selection_strategy_and_selected_critics() -> None:
    state = _make_state()

    result = policy_engine_node(state)

    diagnostics = result.memory_context["policy_engine"]
    assert diagnostics["selection_strategy"] == "top_1"
    assert diagnostics["selected_critics"] == [diagnostics["ranking"][0]["critic_name"]]


def test_does_not_modify_selected_critics_or_policy_decision() -> None:
    """policy_engine_node must not overwrite router_node's territory."""
    state = _make_state()

    result = policy_engine_node(state)

    assert result.selected_critics == []
    assert result.policy_decision is None


def test_does_not_modify_unrelated_state_fields() -> None:
    state = _make_state()

    result = policy_engine_node(state)

    assert result.worker_outputs == []
    assert result.error_features == []
    assert result.critic_feedback == []
    assert result.critic_scores == {}
    assert result.aggregated_quality_score is None
    assert result.correction_history == []
    assert result.final_response is None


def test_preserves_existing_memory_context_keys() -> None:
    state = _make_state()
    state.memory_context = {"existing": "value"}

    result = policy_engine_node(state)

    assert result.memory_context["existing"] == "value"
    assert "policy_engine" in result.memory_context


def test_returns_same_state_instance() -> None:
    state = _make_state()

    result = policy_engine_node(state)

    assert result is state


def test_is_deterministic() -> None:
    def _make_loaded_state() -> AgentState:
        state = _make_state()
        state.error_features = [_high_signal_error_feature()]
        return state

    result_a = policy_engine_node(_make_loaded_state())
    result_b = policy_engine_node(_make_loaded_state())

    assert result_a.memory_context["policy_engine"] == result_b.memory_context["policy_engine"]
