"""Unit tests for `CriticRanking` (app/policy_engine/ranking.py)."""

import pytest

from app.policy_engine.ranking import CriticRanking, RankedCritic


def test_ranking_orders_highest_score_first() -> None:
    ranking = CriticRanking({"A": 0.2, "B": 0.9, "C": 0.5})

    assert ranking.critic_names() == ["B", "C", "A"]


def test_ranked_critics_have_correct_ranks() -> None:
    ranking = CriticRanking({"A": 0.2, "B": 0.9, "C": 0.5})

    ranks = {entry.critic_name: entry.rank for entry in ranking.ranked_critics}
    assert ranks == {"B": 1, "C": 2, "A": 3}


def test_ranked_critics_preserve_scores() -> None:
    ranking = CriticRanking({"A": 0.2, "B": 0.9})

    entries = {entry.critic_name: entry.score for entry in ranking.ranked_critics}
    assert entries == {"A": 0.2, "B": 0.9}


def test_tie_breaking_is_alphabetical() -> None:
    ranking = CriticRanking(
        {"MetaCritic": 0.5, "CodeCritic": 0.5, "LogicCritic": 0.5, "FactCritic": 0.5}
    )

    assert ranking.critic_names() == ["CodeCritic", "FactCritic", "LogicCritic", "MetaCritic"]


def test_tie_breaking_is_deterministic_regardless_of_insertion_order() -> None:
    ranking_a = CriticRanking({"Z": 0.5, "A": 0.5, "M": 0.5})
    ranking_b = CriticRanking({"A": 0.5, "M": 0.5, "Z": 0.5})

    assert ranking_a.critic_names() == ranking_b.critic_names() == ["A", "M", "Z"]


def test_partial_ties_are_broken_alphabetically_within_the_tied_group() -> None:
    ranking = CriticRanking({"B": 0.9, "A": 0.5, "C": 0.5, "D": 0.1})

    assert ranking.critic_names() == ["B", "A", "C", "D"]


def test_top_returns_requested_count() -> None:
    ranking = CriticRanking({"A": 0.2, "B": 0.9, "C": 0.5})

    top_two = ranking.top(2)

    assert [entry.critic_name for entry in top_two] == ["B", "C"]


def test_top_zero_returns_empty_list() -> None:
    ranking = CriticRanking({"A": 0.2, "B": 0.9})

    assert ranking.top(0) == []


def test_top_exceeding_candidate_count_returns_all() -> None:
    ranking = CriticRanking({"A": 0.2, "B": 0.9})

    assert len(ranking.top(10)) == 2


def test_top_rejects_negative_n() -> None:
    ranking = CriticRanking({"A": 0.2})

    with pytest.raises(ValueError):
        ranking.top(-1)


def test_empty_scores_produce_empty_ranking() -> None:
    ranking = CriticRanking({})

    assert ranking.ranked_critics == []
    assert ranking.critic_names() == []
    assert ranking.top(1) == []


def test_single_candidate_ranking() -> None:
    ranking = CriticRanking({"Solo": 0.42})

    assert ranking.critic_names() == ["Solo"]
    assert ranking.ranked_critics[0].rank == 1


def test_score_for_returns_original_score() -> None:
    ranking = CriticRanking({"A": 0.2, "B": 0.9})

    assert ranking.score_for("B") == 0.9


def test_score_for_unknown_critic_raises_key_error() -> None:
    ranking = CriticRanking({"A": 0.2})

    with pytest.raises(KeyError):
        ranking.score_for("NotThere")


def test_as_list_of_dicts_matches_ranked_critics() -> None:
    ranking = CriticRanking({"A": 0.2, "B": 0.9})

    dicts = ranking.as_list_of_dicts()

    assert dicts == [
        {"critic_name": "B", "score": 0.9, "rank": 1},
        {"critic_name": "A", "score": 0.2, "rank": 2},
    ]


def test_ranking_does_not_mutate_input_dict() -> None:
    scores = {"A": 0.2, "B": 0.9}
    original = dict(scores)

    CriticRanking(scores)

    assert scores == original


def test_ranked_critic_is_frozen_dataclass() -> None:
    entry = RankedCritic(critic_name="A", score=0.5, rank=1)

    with pytest.raises(AttributeError):
        entry.score = 0.9  # type: ignore[misc]
