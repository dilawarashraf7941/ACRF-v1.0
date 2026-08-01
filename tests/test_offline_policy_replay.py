"""Integration tests: replaying the real `HeuristicPolicy` and `LinUCBPolicy`
through `ReplayEngine`/`OfflineEvaluator`/`Benchmark`.

`ReplayEngine.replay` (the original, deterministic mode) never calls
`update()` on the policy it replays. For `LinUCBPolicy` to show
genuinely context-sensitive behavior under that mode, most tests below
pre-train a `LinUCBPolicy` instance directly (via `.update()`, outside
`ReplayEngine`) before wrapping it in a `ReplayEngine` — exactly the
workflow `app/evaluation/offline/README.md` documents. `HeuristicPolicy`
has no such training step: it scores critics from nine specifically-named
context features (`app/policy/heuristic_policy.py`) that the
offline-replay `ContextVector` never produces, so it deterministically
selects the same critic (`CodeCritic`, empirically, since ties are
broken alphabetically) for every experience — a documented, expected
consequence of replaying purely from stored `ExperienceRecord`s with no
live `AgentState`.

The last section below (`replay_with_learning`: Sequential Replay
Learning Mode) is the one exception to "`ReplayEngine` never calls
`update()`" — it exercises the engine's additional, explicitly opt-in
`replay_with_learning` method against a real, freshly-constructed
(untrained) `LinUCBPolicy`, demonstrating that training happens
sequentially, in place, as the pass progresses.
"""

from datetime import datetime, timezone

import pytest

from app.evaluation.offline import (
    Benchmark,
    OfflineEvaluator,
    ReplayablePolicy,
    ReplayEngine,
    build_offline_context_vector,
)
from app.experience import ExperienceRecord, InMemoryExperienceRepository
from app.policy import HeuristicPolicy
from app.policy.linucb import LinUCBPolicy
from app.reward import RewardCalculator

_TIMESTAMP = datetime(2024, 1, 1, tzinfo=timezone.utc)
_ALL_CRITICS = ["LogicCritic", "CodeCritic", "FactCritic", "MetaCritic"]
_OTHER_CRITICS = {
    "CodeCritic": ["LogicCritic", "FactCritic", "MetaCritic"],
    "LogicCritic": ["CodeCritic", "FactCritic", "MetaCritic"],
}


_STATE_FEATURES = {
    "task_type": "general",
    "max_iterations": 5,
    "error_feature_count": 0,
    "worker_output_count": 1,
}
"""A fixed, non-degenerate `state_features` snapshot shared by every experience
below, so `build_offline_context_vector` never sees an all-zero context (a
zero context vector carries no signal for LinUCB to train on: `A_inv @ (r *
0) == 0` regardless of `r`). Every experience in a given test uses this same
snapshot, matching this module's design: `LinUCBPolicy`'s trained
*preference* (see `_pretrained_linucb_policy_favoring`) must generalize
identically across every later query context for these tests' assertions
about which critic gets selected to hold.
"""


def _make_experience(
    experience_id: str,
    selected_critics: list[str],
    quality: float = 0.7,
    iterations: int = 1,
    latency: float = 1.0,
) -> ExperienceRecord:
    return ExperienceRecord(
        experience_id=experience_id,
        session_id="session-1",
        task_id="task-1",
        timestamp=_TIMESTAMP,
        state_features=dict(_STATE_FEATURES),
        selected_critics=selected_critics,
        critic_scores={critic: 0.7 for critic in selected_critics},
        aggregated_quality_score=quality,
        iterations=iterations,
        execution_status="completed",
        latency=latency,
    )


def _make_repository() -> InMemoryExperienceRepository:
    repository = InMemoryExperienceRepository()
    repository.add(
        _make_experience("exp-1", ["CodeCritic"], quality=0.9, iterations=1, latency=1.0)
    )
    repository.add(
        _make_experience("exp-2", ["LogicCritic"], quality=0.4, iterations=3, latency=2.0)
    )
    repository.add(
        _make_experience("exp-3", ["CodeCritic"], quality=0.6, iterations=0, latency=0.5)
    )
    repository.add(
        _make_experience(
            "exp-4", ["FactCritic", "MetaCritic"], quality=0.2, iterations=4, latency=3.0
        )
    )
    return repository


def _make_engine(
    repository: InMemoryExperienceRepository, policy: ReplayablePolicy
) -> ReplayEngine:
    return ReplayEngine(repository=repository, policy=policy, reward_calculator=RewardCalculator())


def _heuristic_policy_always_picks_code_critic() -> None:
    """Empirically confirm this test file's premise before relying on it."""
    context = build_offline_context_vector(_make_experience("probe", []))
    decision = HeuristicPolicy().select_action(context, _ALL_CRITICS)
    assert decision.selected_critics == ["CodeCritic"]


# --- HeuristicPolicy replay ---


def test_heuristic_policy_replay_matches_the_code_critic_experiences() -> None:
    _heuristic_policy_always_picks_code_critic()
    repository = _make_repository()
    engine = _make_engine(repository, HeuristicPolicy())

    steps = engine.replay()

    assert {step.experience_id for step in steps} == {"exp-1", "exp-3"}
    assert all(step.selected_critics == ["CodeCritic"] for step in steps)


def test_heuristic_policy_offline_evaluate_produces_replay_result() -> None:
    repository = _make_repository()
    engine = _make_engine(repository, HeuristicPolicy())

    result = OfflineEvaluator().evaluate(engine, policy_name="HeuristicPolicy")

    assert result.policy_name == "HeuristicPolicy"
    assert result.total_experiences == 2
    assert result.critic_selection_frequency == {"CodeCritic": 1.0}


def test_heuristic_policy_replay_is_deterministic() -> None:
    repository = _make_repository()

    engine_a = _make_engine(repository, HeuristicPolicy())
    engine_b = _make_engine(repository, HeuristicPolicy())

    assert engine_a.replay() == engine_b.replay()


def test_heuristic_policy_replay_does_not_mutate_repository_or_records() -> None:
    repository = _make_repository()
    dumps_before = {record.experience_id: record.model_dump() for record in repository.list()}
    engine = _make_engine(repository, HeuristicPolicy())

    engine.replay()

    assert repository.count() == 4
    for record in repository.list():
        assert record.model_dump() == dumps_before[record.experience_id]


# --- LinUCBPolicy replay (pre-trained outside ReplayEngine) ---


def _pretrained_linucb_policy_favoring(critic: str) -> LinUCBPolicy:
    """Train a LinUCBPolicy (via direct `.update()` calls) to prefer `critic`.

    Training happens entirely outside `ReplayEngine`, matching this
    framework's "evaluation never trains" design: `ReplayEngine.replay`
    itself never calls `update`.
    """
    policy = LinUCBPolicy(alpha=0.05)
    warmup_context = build_offline_context_vector(_make_experience("warmup", []))
    for name in [critic, *_OTHER_CRITICS[critic]]:
        policy.update(warmup_context, action=name, reward=0.0)
    for _ in range(8):
        policy.update(warmup_context, action=critic, reward=1.0)
    return policy


def test_linucb_policy_replay_matches_experiences_it_would_select() -> None:
    repository = _make_repository()
    policy = _pretrained_linucb_policy_favoring("CodeCritic")
    engine = _make_engine(repository, policy)

    steps = engine.replay()

    assert {step.experience_id for step in steps} == {"exp-1", "exp-3"}
    assert all(step.selected_critics == ["CodeCritic"] for step in steps)


def test_linucb_policy_trained_toward_a_different_critic_matches_differently() -> None:
    repository = _make_repository()
    policy = _pretrained_linucb_policy_favoring("LogicCritic")
    engine = _make_engine(repository, policy)

    steps = engine.replay()

    assert {step.experience_id for step in steps} == {"exp-2"}


def test_linucb_policy_offline_evaluate_produces_replay_result() -> None:
    repository = _make_repository()
    policy = _pretrained_linucb_policy_favoring("CodeCritic")
    engine = _make_engine(repository, policy)

    result = OfflineEvaluator().evaluate(engine, policy_name="LinUCBPolicy")

    assert result.policy_name == "LinUCBPolicy"
    assert result.total_experiences == 2
    assert result.critic_selection_frequency == {"CodeCritic": 1.0}


def test_linucb_policy_replay_is_deterministic_given_identical_arm_state() -> None:
    repository = _make_repository()
    policy_a = _pretrained_linucb_policy_favoring("CodeCritic")
    policy_b = _pretrained_linucb_policy_favoring("CodeCritic")

    engine_a = _make_engine(repository, policy_a)
    engine_b = _make_engine(repository, policy_b)

    assert engine_a.replay() == engine_b.replay()


def test_linucb_policy_replay_does_not_call_update() -> None:
    repository = _make_repository()
    policy = _pretrained_linucb_policy_favoring("CodeCritic")
    arm_state_before = {
        name: (arm.A.copy(), arm.A_inv.copy(), arm.b.copy()) for name, arm in policy.arms.items()
    }
    engine = _make_engine(repository, policy)

    engine.replay()

    for name, arm in policy.arms.items():
        a_before, a_inv_before, b_before = arm_state_before[name]
        assert (a_before == arm.A).all()
        assert (a_inv_before == arm.A_inv).all()
        assert (b_before == arm.b).all()


# --- Benchmark: comparing two real policies end-to-end ---


def test_benchmark_compares_heuristic_and_linucb_end_to_end() -> None:
    repository = _make_repository()
    evaluator = OfflineEvaluator()

    heuristic_engine = _make_engine(repository, HeuristicPolicy())
    linucb_policy = _pretrained_linucb_policy_favoring("LogicCritic")
    linucb_engine = _make_engine(repository, linucb_policy)

    heuristic_result = evaluator.evaluate(heuristic_engine, policy_name="HeuristicPolicy")
    linucb_result = evaluator.evaluate(linucb_engine, policy_name="LinUCBPolicy")
    benchmark_result = Benchmark().compare(baseline=heuristic_result, candidate=linucb_result)

    assert benchmark_result.baseline_policy == "HeuristicPolicy"
    assert benchmark_result.candidate_policy == "LinUCBPolicy"
    assert benchmark_result.winner in {"HeuristicPolicy", "LinUCBPolicy", "tie"}
    assert benchmark_result.reward_improvement == (
        linucb_result.average_reward - heuristic_result.average_reward
    )


# --- replay_with_learning: Sequential Replay Learning Mode (real LinUCBPolicy) ---


def _make_experience_with_status(
    experience_id: str, selected_critics: list[str], quality: float, status: str
) -> ExperienceRecord:
    return ExperienceRecord(
        experience_id=experience_id,
        session_id="session-1",
        task_id="task-1",
        timestamp=_TIMESTAMP,
        state_features=dict(_STATE_FEATURES),
        selected_critics=selected_critics,
        critic_scores={critic: quality for critic in selected_critics},
        aggregated_quality_score=quality,
        iterations=0,
        execution_status=status,
        latency=0.5,
    )


def test_replay_with_learning_flips_selection_mid_pass_for_untrained_linucb() -> None:
    """A single strongly-negative-reward match is enough to change what an
    untrained LinUCBPolicy selects for every subsequent experience in the
    same `replay_with_learning` call -- demonstrating that the pass is
    genuinely sequential, not a batch computed against one frozen state.
    """
    repository = InMemoryExperienceRepository()
    # 8 CodeCritic experiences with a strongly negative reward (failed, zero quality).
    for i in range(8):
        repository.add(
            _make_experience_with_status(
                f"exp-code-{i}", ["CodeCritic"], quality=0.0, status="failed"
            )
        )
    # Then 4 FactCritic experiences with a strongly positive reward.
    for i in range(4):
        repository.add(
            _make_experience_with_status(
                f"exp-fact-{i}", ["FactCritic"], quality=1.0, status="completed"
            )
        )

    policy = LinUCBPolicy(alpha=0.1)
    engine = _make_engine(repository, policy)

    steps = engine.replay_with_learning()

    # Only the first CodeCritic experience matches (the untrained policy's
    # cold-start tie-break); after training on its negative reward, the
    # policy switches to FactCritic for the rest of the pass, which then
    # matches every remaining FactCritic experience.
    assert [step.experience_id for step in steps] == [
        "exp-code-0",
        "exp-fact-0",
        "exp-fact-1",
        "exp-fact-2",
        "exp-fact-3",
    ]
    assert steps[0].selected_critics == ["CodeCritic"]
    assert all(step.selected_critics == ["FactCritic"] for step in steps[1:])
    assert steps[0].reward < 0.0
    assert all(step.reward > 0.0 for step in steps[1:])


def test_replay_with_learning_trains_the_policy_in_place() -> None:
    repository = InMemoryExperienceRepository()
    repository.add(_make_experience("exp-1", ["CodeCritic"], quality=0.9))

    policy = LinUCBPolicy(alpha=0.1)
    engine = _make_engine(repository, policy)
    arm_before = policy.arms.get("CodeCritic")
    assert arm_before is None  # no arm exists until the policy first sees this action

    engine.replay_with_learning()

    trained_arm = policy.arms["CodeCritic"]
    assert not (trained_arm.b == 0.0).all()


def test_replay_and_replay_with_learning_diverge_from_identical_starting_policies() -> None:
    """`replay()` leaves its policy exactly as constructed; `replay_with_learning()`
    trains its policy as it goes. Two identically-constructed LinUCBPolicy
    instances replayed the two different ways over the same data can end
    up selecting differently, illustrating why the two modes are directly
    comparable but not required to agree.
    """
    repository = InMemoryExperienceRepository()
    for i in range(8):
        repository.add(
            _make_experience_with_status(
                f"exp-code-{i}", ["CodeCritic"], quality=0.0, status="failed"
            )
        )
    for i in range(4):
        repository.add(
            _make_experience_with_status(
                f"exp-fact-{i}", ["FactCritic"], quality=1.0, status="completed"
            )
        )

    plain_engine = _make_engine(repository, LinUCBPolicy(alpha=0.1))
    learning_engine = _make_engine(repository, LinUCBPolicy(alpha=0.1))

    plain_steps = plain_engine.replay()
    learning_steps = learning_engine.replay_with_learning()

    # replay(): the policy never trains, so it ties identically on every
    # experience and only ever matches CodeCritic (its cold-start tie-break).
    assert all(step.selected_critics == ["CodeCritic"] for step in plain_steps)
    assert len(plain_steps) == 8

    # replay_with_learning(): the same starting policy state learns from
    # the first (negative) match and switches to FactCritic thereafter.
    assert len(learning_steps) == 5
    assert learning_steps[-1].selected_critics == ["FactCritic"]


def test_replay_with_learning_does_not_mutate_repository_with_real_linucb() -> None:
    repository = _make_repository()
    dumps_before = {record.experience_id: record.model_dump() for record in repository.list()}
    policy = LinUCBPolicy(alpha=0.5)
    engine = _make_engine(repository, policy)

    engine.replay_with_learning()

    assert repository.count() == 4
    for record in repository.list():
        assert record.model_dump() == dumps_before[record.experience_id]


def test_replay_with_learning_raises_for_heuristic_policy() -> None:
    repository = _make_repository()
    engine = _make_engine(repository, HeuristicPolicy())

    with pytest.raises(AttributeError, match="update"):
        engine.replay_with_learning()
