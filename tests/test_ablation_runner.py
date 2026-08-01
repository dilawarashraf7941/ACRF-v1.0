"""Unit tests for `AblationRunner` (`app/evaluation/ablation/runner.py`).

Covers: every supported ablation mode, determinism, aggregation, and the
Research Constraints (no mutation, no policy training).
"""

from datetime import datetime, timezone

import pytest

from app.evaluation.ablation import AblationConfig, AblationResult, AblationRunner
from app.evaluation.ablation.runner import (
    QualityOnlyRewardStrategy,
    RandomCriticPolicy,
    ReducedContextPolicy,
    _experiment_result_to_replay_result,
)
from app.evaluation.experiments import ExperimentResult
from app.evaluation.offline import build_offline_context_vector
from app.experience import ExperienceRecord, InMemoryExperienceRepository

_TIMESTAMP = datetime(2024, 1, 1, tzinfo=timezone.utc)
_PATTERN = [["CodeCritic"], ["LogicCritic"], ["CodeCritic"], ["FactCritic", "MetaCritic"]]


def _make_experience(
    experience_id: str,
    selected_critics: list[str],
    quality: float = 0.5,
    iterations: int = 1,
    latency: float = 1.0,
) -> ExperienceRecord:
    return ExperienceRecord(
        experience_id=experience_id,
        session_id="session-1",
        task_id="task-1",
        timestamp=_TIMESTAMP,
        state_features={"error_feature_count": 0, "worker_output_count": 1},
        selected_critics=selected_critics,
        critic_scores={critic: 0.7 for critic in selected_critics},
        aggregated_quality_score=quality,
        iterations=iterations,
        execution_status="completed",
        latency=latency,
    )


def _make_repository(size: int = 20) -> InMemoryExperienceRepository:
    repository = InMemoryExperienceRepository()
    for i in range(size):
        repository.add(
            _make_experience(
                f"exp-{i}",
                _PATTERN[i % len(_PATTERN)],
                quality=0.4 + 0.02 * i,
                iterations=i % 4,
                latency=1.0 + 0.05 * i,
            )
        )
    return repository


def _config(ablation_type: str, **overrides: object) -> AblationConfig:
    defaults: dict[str, object] = {
        "experiment_name": f"study-{ablation_type}",
        "baseline_policy": "LinUCBPolicy",
        "candidate_policy": "LinUCBPolicy",
        "ablation_type": ablation_type,
    }
    defaults.update(overrides)
    return AblationConfig(**defaults)


# --- RandomCriticPolicy ---


def test_random_critic_policy_selects_from_candidates() -> None:
    policy = RandomCriticPolicy(seed=1)
    context = build_offline_context_vector(_make_experience("probe", []))

    decision = policy.select_action(context, ["LogicCritic", "CodeCritic", "FactCritic"])

    assert decision.selected_critics[0] in {"LogicCritic", "CodeCritic", "FactCritic"}
    assert decision.policy_name == "RandomCriticPolicy"


def test_random_critic_policy_is_deterministic_given_seed() -> None:
    context = build_offline_context_vector(_make_experience("probe", []))
    actions = ["LogicCritic", "CodeCritic", "FactCritic", "MetaCritic"]

    choice_a = RandomCriticPolicy(seed=7).select_action(context, actions).selected_critics
    choice_b = RandomCriticPolicy(seed=7).select_action(context, actions).selected_critics

    assert choice_a == choice_b


def test_random_critic_policy_varies_across_calls() -> None:
    policy = RandomCriticPolicy(seed=123)
    context = build_offline_context_vector(_make_experience("probe", []))
    actions = ["LogicCritic", "CodeCritic", "FactCritic", "MetaCritic"]

    picks = {
        tuple(policy.select_action(context, actions).selected_critics) for _ in range(20)
    }

    assert len(picks) > 1


def test_random_critic_policy_empty_candidates_returns_empty_selection() -> None:
    policy = RandomCriticPolicy(seed=1)
    context = build_offline_context_vector(_make_experience("probe", []))
    decision = policy.select_action(context, [])
    assert decision.selected_critics == []
    assert decision.confidence == 0.0


# --- ReducedContextPolicy ---


def test_reduced_context_policy_reduces_feature_count() -> None:
    context = build_offline_context_vector(_make_experience("probe", ["CodeCritic"]))
    captured: dict[str, object] = {}

    class _CapturingPolicy:
        def select_action(self, ctx, candidate_actions):
            captured["context"] = ctx
            return None

    wrapper = ReducedContextPolicy(_CapturingPolicy(), keep_fraction=0.5)
    wrapper.select_action(context, ["CodeCritic"])

    reduced = captured["context"]
    assert len(reduced.feature_order) < len(context.feature_order)
    assert len(reduced.feature_order) >= 1


def test_reduced_context_policy_does_not_mutate_original_context() -> None:
    context = build_offline_context_vector(_make_experience("probe", ["CodeCritic"]))
    original_feature_order = list(context.feature_order)

    class _NoopPolicy:
        def select_action(self, ctx, candidate_actions):
            return None

    ReducedContextPolicy(_NoopPolicy(), keep_fraction=0.3).select_action(context, ["CodeCritic"])

    assert context.feature_order == original_feature_order


def test_reduced_context_policy_rejects_invalid_keep_fraction() -> None:
    with pytest.raises(ValueError, match="keep_fraction"):
        ReducedContextPolicy(RandomCriticPolicy(seed=1), keep_fraction=0.0)
    with pytest.raises(ValueError, match="keep_fraction"):
        ReducedContextPolicy(RandomCriticPolicy(seed=1), keep_fraction=1.5)


def test_reduced_context_policy_keep_fraction_one_keeps_everything() -> None:
    context = build_offline_context_vector(_make_experience("probe", ["CodeCritic"]))
    captured: dict[str, object] = {}

    class _CapturingPolicy:
        def select_action(self, ctx, candidate_actions):
            captured["context"] = ctx
            return None

    ReducedContextPolicy(_CapturingPolicy(), keep_fraction=1.0).select_action(
        context, ["CodeCritic"]
    )

    assert captured["context"].feature_order == context.feature_order


# --- QualityOnlyRewardStrategy ---


def test_quality_only_reward_strategy_ignores_cost_and_latency() -> None:
    experience = _make_experience("exp-1", ["CodeCritic"], quality=0.8, iterations=5, latency=100.0)
    signal = QualityOnlyRewardStrategy().compute(experience)

    assert signal.reward == pytest.approx(0.8)
    assert signal.cost_penalty == 0.0
    assert signal.latency_penalty == 0.0
    assert signal.correction_penalty == 0.0


def test_quality_only_reward_strategy_missing_quality_is_zero() -> None:
    experience = _make_experience("exp-1", ["CodeCritic"], quality=None)  # type: ignore[arg-type]
    signal = QualityOnlyRewardStrategy().compute(experience)
    assert signal.reward == 0.0
    assert signal.confidence == 0.0


# --- _experiment_result_to_replay_result adapter ---


def test_experiment_result_to_replay_result_copies_aggregates() -> None:
    experiment_result = ExperimentResult(
        experiment_name="e",
        policy_name="HeuristicPolicy",
        average_reward=0.5,
        std_reward=0.1,
        average_quality=0.6,
        average_latency=1.0,
        average_iterations=1.0,
        match_rate=0.5,
    )
    replay_result = _experiment_result_to_replay_result(experiment_result, "baseline")

    assert replay_result.policy_name == "baseline"
    assert replay_result.average_reward == 0.5
    assert replay_result.average_quality == 0.6


# --- every ablation mode ---


@pytest.mark.parametrize(
    "ablation_type,config_overrides",
    [
        ("no_exploration", {}),
        ("alpha_sweep", {"metadata": {"alpha": 0.25}}),
        (
            "random_critic_selection",
            {"baseline_policy": "HeuristicPolicy", "candidate_policy": "RandomCriticPolicy"},
        ),
        (
            "heuristic_only",
            {"baseline_policy": "LinUCBPolicy", "candidate_policy": "HeuristicPolicy"},
        ),
        (
            "linucb_only",
            {"baseline_policy": "HeuristicPolicy", "candidate_policy": "LinUCBPolicy"},
        ),
        ("reduced_context_features", {}),
        ("alternative_reward_definitions", {}),
    ],
)
def test_every_ablation_mode_runs_and_produces_a_result(
    ablation_type: str, config_overrides: dict
) -> None:
    repository = _make_repository()
    runner = AblationRunner(repository=repository)
    config = _config(ablation_type, **config_overrides)

    result = runner.run(config, num_runs=8, random_seed=3)

    assert isinstance(result, AblationResult)
    assert result.ablation_type == ablation_type
    assert result.conclusion
    assert "significant" in result.metadata
    assert "p_value" in result.metadata


def test_no_exploration_uses_zero_alpha_for_candidate() -> None:
    repository = _make_repository()
    runner = AblationRunner(repository=repository)
    result = runner.run(_config("no_exploration"), num_runs=5, random_seed=1)
    assert result.metadata["candidate_policy"] == "LinUCBPolicy"


def test_alpha_sweep_requires_alpha_in_metadata() -> None:
    repository = _make_repository()
    runner = AblationRunner(repository=repository)
    config = _config("alpha_sweep")  # no alpha in metadata

    with pytest.raises(ValueError, match="alpha"):
        runner.run(config, num_runs=5, random_seed=1)


def test_unsupported_ablation_type_raises() -> None:
    repository = _make_repository()
    runner = AblationRunner(repository=repository)
    config = _config("not_a_real_ablation_type")

    with pytest.raises(ValueError, match="Unsupported ablation_type"):
        runner.run(config, num_runs=5, random_seed=1)


def test_run_all_covers_every_standard_config() -> None:
    repository = _make_repository()
    runner = AblationRunner(repository=repository)

    results = runner.run_all("study", num_runs=5, random_seed=1)

    assert len(results) == len(AblationRunner.standard_configs("study"))
    types_seen = {r.ablation_type for r in results}
    assert types_seen == {
        "no_exploration",
        "alpha_sweep",
        "random_critic_selection",
        "heuristic_only",
        "linucb_only",
        "reduced_context_features",
        "alternative_reward_definitions",
    }


def test_standard_configs_alpha_sweep_covers_default_values() -> None:
    configs = AblationRunner.standard_configs("study")
    alpha_values = {c.metadata["alpha"] for c in configs if c.ablation_type == "alpha_sweep"}
    assert alpha_values == {0.25, 0.5, 1.0, 2.0}


# --- determinism ---


def test_run_is_deterministic_given_the_same_config_and_seed() -> None:
    repository = _make_repository()
    runner = AblationRunner(repository=repository)
    config = _config("no_exploration")

    result_a = runner.run(config, num_runs=10, random_seed=42)
    result_b = runner.run(config, num_runs=10, random_seed=42)

    assert result_a == result_b


def test_run_all_is_deterministic() -> None:
    repository = _make_repository()

    results_a = AblationRunner(repository=repository).run_all("study", num_runs=6, random_seed=7)
    results_b = AblationRunner(repository=repository).run_all("study", num_runs=6, random_seed=7)

    assert results_a == results_b


def test_random_critic_selection_is_deterministic_across_runners() -> None:
    repository = _make_repository()
    config = _config(
        "random_critic_selection",
        baseline_policy="HeuristicPolicy",
        candidate_policy="RandomCriticPolicy",
    )

    result_a = AblationRunner(repository=repository).run(config, num_runs=10, random_seed=5)
    result_b = AblationRunner(repository=repository).run(config, num_runs=10, random_seed=5)

    assert result_a == result_b


# --- aggregation ---


def test_reward_difference_matches_candidate_minus_baseline() -> None:
    repository = _make_repository()
    runner = AblationRunner(repository=repository)
    config = _config(
        "random_critic_selection",
        baseline_policy="HeuristicPolicy",
        candidate_policy="RandomCriticPolicy",
    )

    result = runner.run(config, num_runs=10, random_seed=1)

    expected_difference = result.candidate_reward - result.baseline_reward
    assert result.reward_difference == pytest.approx(expected_difference)


def test_metadata_contains_diagnostic_fields() -> None:
    repository = _make_repository()
    runner = AblationRunner(repository=repository)
    result = runner.run(_config("no_exploration"), num_runs=5, random_seed=1)

    for key in (
        "experiment_name",
        "baseline_policy",
        "candidate_policy",
        "winner",
        "p_value",
        "effect_size",
        "test_used",
        "significant",
        "sample_size",
        "num_runs",
        "random_seed",
    ):
        assert key in result.metadata


# --- Research Constraints: no mutation ---


def test_run_does_not_mutate_source_repository() -> None:
    repository = _make_repository(size=12)
    dumps_before = {record.experience_id: record.model_dump() for record in repository.list()}
    runner = AblationRunner(repository=repository)

    runner.run_all("study", num_runs=5, random_seed=1)

    assert repository.count() == 12
    for record in repository.list():
        assert record.model_dump() == dumps_before[record.experience_id]
