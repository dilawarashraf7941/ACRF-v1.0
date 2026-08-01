"""Unit tests for `MetricsCollector` (app/metrics/collector.py)."""

from datetime import datetime, timezone

from app.experience import ExperienceRecord
from app.metrics import (
    DEFAULT_POLICY_NAME,
    ExecutionMetrics,
    InMemoryMetricsRepository,
    MetricsCollector,
)
from app.reward import RewardSignal
from app.state import AgentState, CorrectionRecord, WorkerOutput


def _make_state(**overrides: object) -> AgentState:
    defaults: dict[str, object] = {"session_id": "s1", "task_id": "t1", "user_query": "q"}
    defaults.update(overrides)
    return AgentState(**defaults)  # type: ignore[arg-type]


def _make_experience(**overrides: object) -> ExperienceRecord:
    defaults: dict[str, object] = {
        "experience_id": "exp-1",
        "session_id": "s1",
        "task_id": "t1",
        "timestamp": datetime.now(timezone.utc),
        "iterations": 0,
        "execution_status": "completed",
    }
    defaults.update(overrides)
    return ExperienceRecord(**defaults)  # type: ignore[arg-type]


def _make_reward(**overrides: object) -> RewardSignal:
    defaults: dict[str, object] = {
        "reward": 0.5,
        "quality_reward": 0.5,
        "efficiency_penalty": 0.0,
        "cost_penalty": 0.0,
        "latency_penalty": 0.0,
        "correction_penalty": 0.0,
        "completion_bonus": 0.2,
        "confidence": 1.0,
        "strategy": "WeightedRewardStrategy",
        "explanation": "test",
    }
    defaults.update(overrides)
    return RewardSignal(**defaults)  # type: ignore[arg-type]


# --- Basic contract ---


def test_collect_returns_execution_metrics() -> None:
    metrics = MetricsCollector().collect(_make_state(), _make_experience(), _make_reward())

    assert isinstance(metrics, ExecutionMetrics)


def test_execution_id_comes_from_experience() -> None:
    experience = _make_experience(experience_id="unique-exp-id")

    metrics = MetricsCollector().collect(_make_state(), experience, _make_reward())

    assert metrics.execution_id == "unique-exp-id"


def test_reward_comes_from_reward_signal() -> None:
    reward = _make_reward(reward=0.87)

    metrics = MetricsCollector().collect(_make_state(), _make_experience(), reward)

    assert metrics.reward == 0.87


def test_copies_fields_directly_from_experience() -> None:
    experience = _make_experience(
        aggregated_quality_score=0.6,
        iterations=3,
        latency=1.2,
        estimated_cost=0.05,
        selected_critics=["LogicCritic", "CodeCritic"],
        execution_status="completed",
    )

    metrics = MetricsCollector().collect(_make_state(), experience, _make_reward())

    assert metrics.aggregated_quality_score == 0.6
    assert metrics.iterations == 3
    assert metrics.latency == 1.2
    assert metrics.estimated_cost == 0.05
    assert metrics.selected_critics == ["LogicCritic", "CodeCritic"]
    assert metrics.execution_status == "completed"


def test_timestamp_comes_from_experience() -> None:
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    experience = _make_experience(timestamp=ts)

    metrics = MetricsCollector().collect(_make_state(), experience, _make_reward())

    assert metrics.timestamp == ts


# --- correction_applied ---


def test_correction_applied_false_when_no_correction_history() -> None:
    state = _make_state()
    assert state.correction_history == []

    metrics = MetricsCollector().collect(state, _make_experience(), _make_reward())

    assert metrics.correction_applied is False


def test_correction_applied_true_when_correction_history_present() -> None:
    state = _make_state()
    state.correction_history = [
        CorrectionRecord(
            iteration=0, description="placeholder correction", applied_by="self_correction"
        )
    ]

    metrics = MetricsCollector().collect(state, _make_experience(), _make_reward())

    assert metrics.correction_applied is True


def test_correction_applied_reflects_history_even_when_correction_decision_says_no() -> None:
    """`correction_applied` must reflect actual correction_history, not just
    the latest correction_decision (which could be stale for multi-iteration runs).
    """
    state = _make_state()
    state.correction_history = [
        CorrectionRecord(iteration=0, description="applied earlier", applied_by="self_correction")
    ]
    experience = _make_experience(correction_decision={"should_correct": False})

    metrics = MetricsCollector().collect(state, experience, _make_reward())

    assert metrics.correction_applied is True


# --- policy tag ---


def test_metadata_policy_defaults_to_heuristic_policy() -> None:
    metrics = MetricsCollector().collect(_make_state(), _make_experience(), _make_reward())

    assert metrics.metadata["policy"] == DEFAULT_POLICY_NAME
    assert DEFAULT_POLICY_NAME == "HeuristicPolicy"


def test_metadata_policy_reads_published_policy_name() -> None:
    state = _make_state()
    state.memory_context = {"policy_engine": {"policy_name": "ContextualBandit"}}

    metrics = MetricsCollector().collect(state, _make_experience(), _make_reward())

    assert metrics.metadata["policy"] == "ContextualBandit"


def test_metadata_policy_falls_back_when_policy_engine_entry_malformed() -> None:
    state = _make_state()
    state.memory_context = {"policy_engine": "not_a_dict"}

    metrics = MetricsCollector().collect(state, _make_experience(), _make_reward())

    assert metrics.metadata["policy"] == DEFAULT_POLICY_NAME


def test_metadata_includes_reward_strategy() -> None:
    reward = _make_reward(strategy="CustomStrategy")

    metrics = MetricsCollector().collect(_make_state(), _make_experience(), reward)

    assert metrics.metadata["reward_strategy"] == "CustomStrategy"


# --- Read-only behavior ---


def test_collect_does_not_mutate_state() -> None:
    state = _make_state()
    state.worker_outputs = [WorkerOutput(worker_id="w")]
    snapshot = state.model_dump(mode="json")

    MetricsCollector().collect(state, _make_experience(), _make_reward())

    assert state.model_dump(mode="json") == snapshot


def test_collect_does_not_mutate_experience_or_reward() -> None:
    experience = _make_experience()
    reward = _make_reward()
    experience_snapshot = experience.model_dump(mode="json")
    reward_snapshot = reward.model_dump(mode="json")

    MetricsCollector().collect(_make_state(), experience, reward)

    assert experience.model_dump(mode="json") == experience_snapshot
    assert reward.model_dump(mode="json") == reward_snapshot


# --- Dependency injection into a repository ---


def test_collect_without_repository_does_not_store_anything() -> None:
    MetricsCollector(repository=None).collect(_make_state(), _make_experience(), _make_reward())
    # Nothing to assert against directly; confirms no exception without a repository.


def test_collect_with_injected_repository_stores_the_record() -> None:
    repository = InMemoryMetricsRepository()
    collector = MetricsCollector(repository=repository)

    metrics = collector.collect(_make_state(), _make_experience(), _make_reward())

    assert repository.count() == 1
    assert repository.list() == [metrics]


# --- Determinism ---


def test_collect_is_deterministic() -> None:
    state = _make_state()
    experience = _make_experience(aggregated_quality_score=0.5)
    reward = _make_reward(reward=0.5)

    metrics_a = MetricsCollector().collect(state, experience, reward)
    metrics_b = MetricsCollector().collect(state, experience, reward)

    assert metrics_a == metrics_b
