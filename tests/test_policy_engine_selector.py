"""Unit tests for `CriticSelector` (app/policy_engine/selector.py)."""

import pytest

from app.policy_engine.ranking import CriticRanking
from app.policy_engine.selector import CriticSelector, SelectionStrategy


@pytest.fixture
def ranking() -> CriticRanking:
    return CriticRanking({"A": 0.2, "B": 0.9, "C": 0.5, "D": 0.5})


# --- Top-1 ---


def test_select_top_1_returns_single_highest(ranking: CriticRanking) -> None:
    assert CriticSelector().select_top_1(ranking) == ["B"]


def test_select_top_1_on_empty_ranking_returns_empty_list() -> None:
    assert CriticSelector().select_top_1(CriticRanking({})) == []


def test_select_via_dispatch_top_1(ranking: CriticRanking) -> None:
    assert CriticSelector().select(ranking, SelectionStrategy.TOP_1) == ["B"]


# --- Top-K ---


def test_select_top_k_returns_k_highest(ranking: CriticRanking) -> None:
    assert CriticSelector().select_top_k(ranking, 2) == ["B", "C"]


def test_select_top_k_zero_returns_empty_list(ranking: CriticRanking) -> None:
    assert CriticSelector().select_top_k(ranking, 0) == []


def test_select_top_k_exceeding_size_returns_all(ranking: CriticRanking) -> None:
    assert CriticSelector().select_top_k(ranking, 100) == ["B", "C", "D", "A"]


def test_select_top_k_rejects_negative_k(ranking: CriticRanking) -> None:
    with pytest.raises(ValueError):
        CriticSelector().select_top_k(ranking, -1)


def test_select_via_dispatch_top_k_requires_k(ranking: CriticRanking) -> None:
    with pytest.raises(ValueError):
        CriticSelector().select(ranking, SelectionStrategy.TOP_K)


def test_select_via_dispatch_top_k(ranking: CriticRanking) -> None:
    assert CriticSelector().select(ranking, SelectionStrategy.TOP_K, k=3) == ["B", "C", "D"]


# --- Threshold ---


def test_select_by_threshold_returns_matching_critics(ranking: CriticRanking) -> None:
    assert CriticSelector().select_by_threshold(ranking, 0.5) == ["B", "C", "D"]


def test_select_by_threshold_is_inclusive_of_boundary(ranking: CriticRanking) -> None:
    assert "C" in CriticSelector().select_by_threshold(ranking, 0.5)


def test_select_by_threshold_above_all_scores_returns_empty(ranking: CriticRanking) -> None:
    assert CriticSelector().select_by_threshold(ranking, 0.99) == []


def test_select_by_threshold_below_all_scores_returns_everyone(ranking: CriticRanking) -> None:
    assert CriticSelector().select_by_threshold(ranking, 0.0) == ["B", "C", "D", "A"]


def test_select_via_dispatch_threshold_requires_threshold(ranking: CriticRanking) -> None:
    with pytest.raises(ValueError):
        CriticSelector().select(ranking, SelectionStrategy.THRESHOLD)


def test_select_via_dispatch_threshold(ranking: CriticRanking) -> None:
    assert CriticSelector().select(ranking, SelectionStrategy.THRESHOLD, threshold=0.6) == ["B"]


# --- Determinism ---


def test_selection_is_deterministic_across_calls(ranking: CriticRanking) -> None:
    selector = CriticSelector()

    assert selector.select_top_k(ranking, 3) == selector.select_top_k(ranking, 3)
    assert selector.select_by_threshold(ranking, 0.5) == selector.select_by_threshold(ranking, 0.5)


def test_selection_preserves_ranking_order_including_ties() -> None:
    tied = CriticRanking({"MetaCritic": 0.5, "CodeCritic": 0.5, "LogicCritic": 0.5})

    assert CriticSelector().select_top_k(tied, 3) == ["CodeCritic", "LogicCritic", "MetaCritic"]


# --- Invalid strategy ---


def test_select_dispatch_rejects_unknown_strategy(ranking: CriticRanking) -> None:
    with pytest.raises(ValueError):
        CriticSelector().select(ranking, "not_a_real_strategy")  # type: ignore[arg-type]
