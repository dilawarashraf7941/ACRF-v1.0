"""Unit tests for `ReplayEngine` and its helpers (`app/evaluation/offline/replay.py`).

Covers: single experience, multiple experiences, empty repository,
no-mutation of the source repository/records, and deterministic replay.
`HeuristicPolicy`/`LinUCBPolicy`-specific replay behavior is covered in
`tests/test_offline_policy_replay.py`.
"""

from datetime import datetime, timezone

import pytest

from app.context import ContextVector
from app.evaluation.offline.models import ReplayStep
from app.evaluation.offline.replay import (
    DEFAULT_CANDIDATE_CRITICS,
    ReplayEngine,
    TrainablePolicy,
    _extract_selected_critics,
    build_offline_context_vector,
)
from app.experience import ExperienceRecord, InMemoryExperienceRepository
from app.reward import RewardCalculator

_TIMESTAMP = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _make_experience(
    experience_id: str = "exp-1",
    selected_critics: list[str] | None = None,
    quality: float | None = 0.8,
    iterations: int = 1,
    latency: float | None = 1.0,
    status: str = "completed",
) -> ExperienceRecord:
    critics = selected_critics if selected_critics is not None else ["CodeCritic"]
    return ExperienceRecord(
        experience_id=experience_id,
        session_id="session-1",
        task_id="task-1",
        timestamp=_TIMESTAMP,
        state_features={"error_feature_count": 0, "worker_output_count": 1},
        selected_critics=critics,
        critic_scores={critic: 0.7 for critic in critics},
        aggregated_quality_score=quality,
        iterations=iterations,
        execution_status=status,
        latency=latency,
    )


class _AlwaysSelectPolicy:
    """A minimal `ReplayablePolicy`: always selects a fixed critic, ignoring context."""

    def __init__(self, critic: str) -> None:
        self._critic = critic
        self.calls: list[tuple[ContextVector, list[str]]] = []

    def select_action(self, context: ContextVector, candidate_actions: list[str]) -> object:
        self.calls.append((context, list(candidate_actions)))

        class _Decision:
            selected_critics = [self._critic]

        return _Decision()


class _SingleActionPolicy:
    """A minimal `ReplayablePolicy` whose decision exposes `selected_action`, not a list."""

    def __init__(self, critic: str) -> None:
        self._critic = critic

    def select_action(self, context: ContextVector, candidate_actions: list[str]) -> object:
        class _Decision:
            selected_action = self._critic

        return _Decision()


class _EmptySelectionPolicy:
    """A `ReplayablePolicy` that selects nothing."""

    def select_action(self, context: ContextVector, candidate_actions: list[str]) -> object:
        class _Decision:
            selected_critics: list[str] = []

        return _Decision()


class _NoSelectionAttributePolicy:
    """A `ReplayablePolicy` whose decision exposes neither expected attribute."""

    def select_action(self, context: ContextVector, candidate_actions: list[str]) -> object:
        return object()


class _TrainableStubPolicy:
    """A minimal `TrainablePolicy`: selects a fixed critic and records every `update` call."""

    def __init__(self, critic: str) -> None:
        self._critic = critic
        self.update_calls: list[tuple[ContextVector, str, float]] = []

    def select_action(self, context: ContextVector, candidate_actions: list[str]) -> object:
        class _Decision:
            selected_critics = [self._critic]

        return _Decision()

    def update(self, context: ContextVector, action: str, reward: float) -> None:
        self.update_calls.append((context, action, reward))


# --- build_offline_context_vector ---


def test_build_offline_context_vector_returns_context_vector() -> None:
    context = build_offline_context_vector(_make_experience())
    assert isinstance(context, ContextVector)


def test_build_offline_context_vector_sets_source_execution_id() -> None:
    context = build_offline_context_vector(_make_experience(experience_id="exp-42"))
    assert context.source_execution_id == "exp-42"


def test_build_offline_context_vector_uses_experience_timestamp() -> None:
    context = build_offline_context_vector(_make_experience())
    assert context.timestamp == _TIMESTAMP


def test_build_offline_context_vector_is_deterministic() -> None:
    experience = _make_experience()
    first = build_offline_context_vector(experience)
    second = build_offline_context_vector(experience)
    assert first == second


def test_build_offline_context_vector_context_id_differs_by_experience_id() -> None:
    context_a = build_offline_context_vector(_make_experience(experience_id="exp-a"))
    context_b = build_offline_context_vector(_make_experience(experience_id="exp-b"))
    assert context_a.context_id != context_b.context_id


def test_build_offline_context_vector_does_not_mutate_experience() -> None:
    experience = _make_experience()
    dump_before = experience.model_dump()
    build_offline_context_vector(experience)
    assert experience.model_dump() == dump_before


def test_build_offline_context_vector_handles_missing_state_features() -> None:
    experience = _make_experience().model_copy(update={"state_features": {}})
    context = build_offline_context_vector(experience)
    assert context.features["is_code_task"] == 0.0
    assert context.features["has_task_type"] == 0.0
    assert context.features["plan_complexity"] == 0.0
    assert context.features["max_iterations"] == 0.0


def test_build_offline_context_vector_flags_code_task_type() -> None:
    experience = _make_experience().model_copy(
        update={"state_features": {"task_type": "code"}}
    )
    context = build_offline_context_vector(experience)
    assert context.features["is_code_task"] == 1.0
    assert context.features["has_task_type"] == 1.0


def test_build_offline_context_vector_non_code_task_type_is_not_flagged() -> None:
    experience = _make_experience().model_copy(
        update={"state_features": {"task_type": "general"}}
    )
    context = build_offline_context_vector(experience)
    assert context.features["is_code_task"] == 0.0
    assert context.features["has_task_type"] == 1.0


def test_build_offline_context_vector_plan_complexity_from_decomposition() -> None:
    experience = _make_experience().model_copy(
        update={"state_features": {"planner_output": {"decomposition": ["a", "b"]}}}
    )
    context = build_offline_context_vector(experience)
    assert context.features["plan_complexity"] == pytest.approx(2 / 5)


def test_build_offline_context_vector_normalizes_max_iterations() -> None:
    experience = _make_experience().model_copy(
        update={"state_features": {"max_iterations": 10}}
    )
    context = build_offline_context_vector(experience)
    assert context.features["max_iterations"] == pytest.approx((10 - 1) / (20 - 1))


def test_build_offline_context_vector_excludes_outcome_and_action_derived_fields() -> None:
    """Regression guard: none of the old, leaky feature names may reappear.

    Every name below reflects the episode's outcome, the action actually
    taken, or a count that can change across an episode's iterations —
    see `build_offline_context_vector`'s docstring for the full audit.
    """
    context = build_offline_context_vector(_make_experience())
    leaky_names = {
        "aggregated_quality_score",
        "iterations",
        "latency",
        "estimated_cost",
        "critic_score_count",
        "critic_score_average",
        "selected_critic_count",
        "has_correction_decision",
        "error_feature_count",
        "worker_output_count",
        "is_completed",
    }
    assert leaky_names.isdisjoint(context.features.keys())


def test_build_offline_context_vector_feature_order_matches_features_keys() -> None:
    context = build_offline_context_vector(_make_experience())
    assert context.feature_order == list(context.features.keys())


# --- _extract_selected_critics ---


def test_extract_selected_critics_from_selected_critics_attribute() -> None:
    class _Decision:
        selected_critics = ["LogicCritic", "CodeCritic"]

    assert _extract_selected_critics(_Decision()) == ["LogicCritic", "CodeCritic"]


def test_extract_selected_critics_from_selected_action_attribute() -> None:
    class _Decision:
        selected_action = "MetaCritic"

    assert _extract_selected_critics(_Decision()) == ["MetaCritic"]


def test_extract_selected_critics_prefers_selected_critics_when_both_present() -> None:
    class _Decision:
        selected_critics = ["LogicCritic"]
        selected_action = "CodeCritic"

    assert _extract_selected_critics(_Decision()) == ["LogicCritic"]


def test_extract_selected_critics_raises_type_error_when_neither_present() -> None:
    with pytest.raises(TypeError, match="neither"):
        _extract_selected_critics(object())


# --- ReplayEngine: single experience ---


def test_replay_single_matching_experience_produces_one_step() -> None:
    repository = InMemoryExperienceRepository()
    repository.add(_make_experience(experience_id="exp-1", selected_critics=["CodeCritic"]))
    engine = ReplayEngine(
        repository=repository,
        policy=_AlwaysSelectPolicy("CodeCritic"),
        reward_calculator=RewardCalculator(),
    )

    steps = engine.replay()

    assert len(steps) == 1
    assert isinstance(steps[0], ReplayStep)
    assert steps[0].experience_id == "exp-1"
    assert steps[0].selected_critics == ["CodeCritic"]


def test_replay_single_non_matching_experience_produces_no_steps() -> None:
    repository = InMemoryExperienceRepository()
    repository.add(_make_experience(experience_id="exp-1", selected_critics=["LogicCritic"]))
    engine = ReplayEngine(
        repository=repository,
        policy=_AlwaysSelectPolicy("CodeCritic"),
        reward_calculator=RewardCalculator(),
    )

    assert engine.replay() == []


def test_replay_step_reward_comes_from_reward_calculator() -> None:
    repository = InMemoryExperienceRepository()
    experience = _make_experience(experience_id="exp-1", selected_critics=["CodeCritic"])
    repository.add(experience)
    reward_calculator = RewardCalculator()
    engine = ReplayEngine(
        repository=repository,
        policy=_AlwaysSelectPolicy("CodeCritic"),
        reward_calculator=reward_calculator,
    )

    steps = engine.replay()

    expected_reward = reward_calculator.calculate(experience).reward
    assert steps[0].reward == expected_reward


# --- ReplayEngine: multiple experiences ---


def test_replay_multiple_experiences_only_counts_matches() -> None:
    repository = InMemoryExperienceRepository()
    repository.add(_make_experience(experience_id="exp-1", selected_critics=["CodeCritic"]))
    repository.add(_make_experience(experience_id="exp-2", selected_critics=["LogicCritic"]))
    repository.add(_make_experience(experience_id="exp-3", selected_critics=["CodeCritic"]))
    engine = ReplayEngine(
        repository=repository,
        policy=_AlwaysSelectPolicy("CodeCritic"),
        reward_calculator=RewardCalculator(),
    )

    steps = engine.replay()

    assert [step.experience_id for step in steps] == ["exp-1", "exp-3"]


def test_replay_preserves_repository_list_order() -> None:
    repository = InMemoryExperienceRepository()
    for i in range(5):
        repository.add(
            _make_experience(experience_id=f"exp-{i}", selected_critics=["CodeCritic"])
        )
    engine = ReplayEngine(
        repository=repository,
        policy=_AlwaysSelectPolicy("CodeCritic"),
        reward_calculator=RewardCalculator(),
    )

    steps = engine.replay()

    assert [step.experience_id for step in steps] == [f"exp-{i}" for i in range(5)]


# --- ReplayEngine: empty repository ---


def test_replay_empty_repository_returns_empty_list() -> None:
    repository = InMemoryExperienceRepository()
    engine = ReplayEngine(
        repository=repository,
        policy=_AlwaysSelectPolicy("CodeCritic"),
        reward_calculator=RewardCalculator(),
    )

    assert engine.replay() == []


# --- ReplayEngine: selection normalization edge cases ---


def test_replay_uses_selected_action_style_decisions() -> None:
    repository = InMemoryExperienceRepository()
    repository.add(_make_experience(experience_id="exp-1", selected_critics=["MetaCritic"]))
    engine = ReplayEngine(
        repository=repository,
        policy=_SingleActionPolicy("MetaCritic"),
        reward_calculator=RewardCalculator(),
    )

    steps = engine.replay()

    assert len(steps) == 1
    assert steps[0].selected_critics == ["MetaCritic"]


def test_replay_skips_experiences_when_policy_selects_nothing() -> None:
    repository = InMemoryExperienceRepository()
    repository.add(_make_experience(experience_id="exp-1", selected_critics=[]))
    engine = ReplayEngine(
        repository=repository, policy=_EmptySelectionPolicy(), reward_calculator=RewardCalculator()
    )

    assert engine.replay() == []


def test_replay_raises_type_error_for_unrecognized_decision_shape() -> None:
    repository = InMemoryExperienceRepository()
    repository.add(_make_experience(experience_id="exp-1"))
    engine = ReplayEngine(
        repository=repository,
        policy=_NoSelectionAttributePolicy(),
        reward_calculator=RewardCalculator(),
    )

    with pytest.raises(TypeError):
        engine.replay()


# --- ReplayEngine: no mutation ---


def test_replay_does_not_mutate_repository() -> None:
    repository = InMemoryExperienceRepository()
    repository.add(_make_experience(experience_id="exp-1", selected_critics=["CodeCritic"]))
    engine = ReplayEngine(
        repository=repository,
        policy=_AlwaysSelectPolicy("CodeCritic"),
        reward_calculator=RewardCalculator(),
    )

    engine.replay()

    assert repository.count() == 1
    assert repository.get("exp-1") is not None


def test_replay_does_not_add_new_experiences() -> None:
    repository = InMemoryExperienceRepository()
    repository.add(_make_experience(experience_id="exp-1", selected_critics=["CodeCritic"]))
    engine = ReplayEngine(
        repository=repository,
        policy=_AlwaysSelectPolicy("CodeCritic"),
        reward_calculator=RewardCalculator(),
    )

    engine.replay()
    engine.replay()

    assert repository.count() == 1


# --- ReplayEngine: candidate actions ---


def test_default_candidate_actions_are_passed_to_policy() -> None:
    repository = InMemoryExperienceRepository()
    repository.add(_make_experience(experience_id="exp-1", selected_critics=["CodeCritic"]))
    policy = _AlwaysSelectPolicy("CodeCritic")
    engine = ReplayEngine(
        repository=repository, policy=policy, reward_calculator=RewardCalculator()
    )

    engine.replay()

    assert policy.calls[0][1] == list(DEFAULT_CANDIDATE_CRITICS)


def test_custom_candidate_actions_are_passed_to_policy() -> None:
    repository = InMemoryExperienceRepository()
    repository.add(_make_experience(experience_id="exp-1", selected_critics=["CodeCritic"]))
    policy = _AlwaysSelectPolicy("CodeCritic")
    engine = ReplayEngine(
        repository=repository,
        policy=policy,
        reward_calculator=RewardCalculator(),
        candidate_actions=["CodeCritic", "LogicCritic"],
    )

    engine.replay()

    assert policy.calls[0][1] == ["CodeCritic", "LogicCritic"]
    assert engine.candidate_actions == ["CodeCritic", "LogicCritic"]


def test_repository_property_exposes_injected_repository() -> None:
    repository = InMemoryExperienceRepository()
    engine = ReplayEngine(
        repository=repository,
        policy=_AlwaysSelectPolicy("CodeCritic"),
        reward_calculator=RewardCalculator(),
    )

    assert engine.repository is repository


# --- ReplayEngine: deterministic replay ---


def test_replay_is_deterministic_across_independent_engines() -> None:
    repository = InMemoryExperienceRepository()
    repository.add(_make_experience(experience_id="exp-1", selected_critics=["CodeCritic"]))
    repository.add(_make_experience(experience_id="exp-2", selected_critics=["LogicCritic"]))

    engine_a = ReplayEngine(
        repository=repository,
        policy=_AlwaysSelectPolicy("CodeCritic"),
        reward_calculator=RewardCalculator(),
    )
    engine_b = ReplayEngine(
        repository=repository,
        policy=_AlwaysSelectPolicy("CodeCritic"),
        reward_calculator=RewardCalculator(),
    )

    assert engine_a.replay() == engine_b.replay()


def test_replay_called_twice_on_same_engine_is_identical() -> None:
    repository = InMemoryExperienceRepository()
    repository.add(_make_experience(experience_id="exp-1", selected_critics=["CodeCritic"]))
    engine = ReplayEngine(
        repository=repository,
        policy=_AlwaysSelectPolicy("CodeCritic"),
        reward_calculator=RewardCalculator(),
    )

    assert engine.replay() == engine.replay()


# --- ReplayEngine.replay_with_learning: Sequential Replay Learning Mode ---
#
# `replay()` itself is not touched anywhere in this section; these tests
# exercise the additional `replay_with_learning` method only, using the
# `_TrainableStubPolicy` helper (defined above, alongside the other stub
# policies). LinUCBPolicy-specific, realistic sequential-learning
# behavior (the policy's own selection changing mid-pass) is covered in
# `tests/test_offline_policy_replay.py`.


def test_replay_with_learning_raises_for_non_trainable_policy() -> None:
    repository = InMemoryExperienceRepository()
    repository.add(_make_experience(experience_id="exp-1", selected_critics=["CodeCritic"]))
    engine = ReplayEngine(
        repository=repository,
        policy=_AlwaysSelectPolicy("CodeCritic"),  # no update() method
        reward_calculator=RewardCalculator(),
    )

    with pytest.raises(AttributeError, match="update"):
        engine.replay_with_learning()


def test_replay_with_learning_raises_before_replaying_anything() -> None:
    repository = InMemoryExperienceRepository()
    repository.add(_make_experience(experience_id="exp-1", selected_critics=["CodeCritic"]))
    policy = _AlwaysSelectPolicy("CodeCritic")
    engine = ReplayEngine(
        repository=repository, policy=policy, reward_calculator=RewardCalculator()
    )

    with pytest.raises(AttributeError):
        engine.replay_with_learning()

    assert policy.calls == []


def test_replay_with_learning_returns_same_shape_as_replay() -> None:
    repository = InMemoryExperienceRepository()
    repository.add(_make_experience(experience_id="exp-1", selected_critics=["CodeCritic"]))
    policy = _TrainableStubPolicy("CodeCritic")
    engine = ReplayEngine(
        repository=repository, policy=policy, reward_calculator=RewardCalculator()
    )

    steps = engine.replay_with_learning()

    assert len(steps) == 1
    assert isinstance(steps[0], ReplayStep)
    assert steps[0].experience_id == "exp-1"
    assert steps[0].selected_critics == ["CodeCritic"]


def test_replay_with_learning_calls_update_for_every_match() -> None:
    repository = InMemoryExperienceRepository()
    repository.add(_make_experience(experience_id="exp-1", selected_critics=["CodeCritic"]))
    repository.add(_make_experience(experience_id="exp-2", selected_critics=["CodeCritic"]))
    policy = _TrainableStubPolicy("CodeCritic")
    engine = ReplayEngine(
        repository=repository, policy=policy, reward_calculator=RewardCalculator()
    )

    steps = engine.replay_with_learning()

    assert len(policy.update_calls) == 2
    for step, (context, action, reward) in zip(steps, policy.update_calls, strict=True):
        assert action == "CodeCritic"
        assert reward == step.reward
        assert context.context_id == step.context_id


def test_replay_with_learning_skips_non_matching_experiences_without_training() -> None:
    repository = InMemoryExperienceRepository()
    repository.add(_make_experience(experience_id="exp-1", selected_critics=["LogicCritic"]))
    policy = _TrainableStubPolicy("CodeCritic")  # never selects LogicCritic
    engine = ReplayEngine(
        repository=repository, policy=policy, reward_calculator=RewardCalculator()
    )

    steps = engine.replay_with_learning()

    assert steps == []
    assert policy.update_calls == []


def test_replay_with_learning_empty_repository_returns_empty_list() -> None:
    repository = InMemoryExperienceRepository()
    policy = _TrainableStubPolicy("CodeCritic")
    engine = ReplayEngine(
        repository=repository, policy=policy, reward_calculator=RewardCalculator()
    )

    assert engine.replay_with_learning() == []
    assert policy.update_calls == []


def test_replay_with_learning_updates_on_every_selected_critic_for_multi_critic_matches() -> None:
    repository = InMemoryExperienceRepository()
    repository.add(
        _make_experience(experience_id="exp-1", selected_critics=["LogicCritic", "MetaCritic"])
    )

    class _MultiCriticPolicy:
        def __init__(self) -> None:
            self.update_calls: list[tuple[str, float]] = []

        def select_action(self, context: ContextVector, candidate_actions: list[str]) -> object:
            class _Decision:
                selected_critics = ["LogicCritic", "MetaCritic"]

            return _Decision()

        def update(self, context: ContextVector, action: str, reward: float) -> None:
            self.update_calls.append((action, reward))

    policy = _MultiCriticPolicy()
    engine = ReplayEngine(
        repository=repository, policy=policy, reward_calculator=RewardCalculator()
    )

    steps = engine.replay_with_learning()

    assert len(steps) == 1
    assert {action for action, _ in policy.update_calls} == {"LogicCritic", "MetaCritic"}


def test_replay_with_learning_does_not_mutate_repository_or_records() -> None:
    repository = InMemoryExperienceRepository()
    repository.add(_make_experience(experience_id="exp-1", selected_critics=["CodeCritic"]))
    dump_before = repository.get("exp-1").model_dump()
    policy = _TrainableStubPolicy("CodeCritic")
    engine = ReplayEngine(
        repository=repository, policy=policy, reward_calculator=RewardCalculator()
    )

    engine.replay_with_learning()

    assert repository.count() == 1
    assert repository.get("exp-1").model_dump() == dump_before


def test_replay_with_learning_is_deterministic_given_fresh_identical_policies() -> None:
    repository = InMemoryExperienceRepository()
    repository.add(_make_experience(experience_id="exp-1", selected_critics=["CodeCritic"]))
    repository.add(_make_experience(experience_id="exp-2", selected_critics=["CodeCritic"]))

    engine_a = ReplayEngine(
        repository=repository,
        policy=_TrainableStubPolicy("CodeCritic"),
        reward_calculator=RewardCalculator(),
    )
    engine_b = ReplayEngine(
        repository=repository,
        policy=_TrainableStubPolicy("CodeCritic"),
        reward_calculator=RewardCalculator(),
    )

    assert engine_a.replay_with_learning() == engine_b.replay_with_learning()


def test_replay_does_not_expose_or_require_update_on_the_policy() -> None:
    """`replay()` must remain usable with a policy that has no `update` at all."""
    repository = InMemoryExperienceRepository()
    repository.add(_make_experience(experience_id="exp-1", selected_critics=["CodeCritic"]))
    engine = ReplayEngine(
        repository=repository,
        policy=_AlwaysSelectPolicy("CodeCritic"),
        reward_calculator=RewardCalculator(),
    )

    steps = engine.replay()

    assert len(steps) == 1


def test_trainable_policy_protocol_recognizes_a_conforming_stub() -> None:
    policy = _TrainableStubPolicy("CodeCritic")
    assert isinstance(policy, TrainablePolicy)


def test_trainable_policy_protocol_rejects_a_non_trainable_stub() -> None:
    policy = _AlwaysSelectPolicy("CodeCritic")
    assert not isinstance(policy, TrainablePolicy)
