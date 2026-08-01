# Ablation Study Framework

Systematically compares system variants — **using the existing
evaluation pipeline only**. No graph, replay, experiment, statistics, or
reward logic is reimplemented anywhere in this module.

> **Scope:** orchestration and reporting only. `app/graph`,
> `app/router`, `app/policy_engine`, `app/evaluation/offline/replay.py`
> (`ReplayEngine`), `app/evaluation/experiments`'s replay/aggregation
> mechanics (`ExperimentRunner`), `app/reward`, `app/experience`, and
> `app/context` are all used exactly as they are. No PPO, no
> reinforcement learning.

## Files

| File | Responsibility |
|---|---|
| `models.py` | `AblationConfig`, `AblationResult`. |
| `runner.py` | `AblationRunner` and three small, self-contained implementations of *existing* extension points. |
| `report.py` | `AblationReportGenerator` — Markdown (summary table, ranking, best/worst, key observations), CSV, JSON. |

## Everything is composition, not new infrastructure

`AblationRunner.run` never touches replay, resampling, reward
computation, or hypothesis testing directly. For each ablation it:

1. Builds two `app.evaluation.experiments.ExperimentConfig`s (baseline,
   candidate) and replays each through an
   `app.evaluation.experiments.ExperimentRunner` — which itself replays
   via `app.evaluation.offline.ReplayEngine` and aggregates via
   `OfflineEvaluator`. **Unmodified.**
2. Reshapes the two resulting `ExperimentResult`s into `ReplayResult`s
   (pure field copying — see `_experiment_result_to_replay_result`) and
   hands them to `app.evaluation.offline.Benchmark.compare` for the four
   delta fields (`reward_difference`, `quality_difference`,
   `latency_difference`, `iteration_difference`) and the winner.
   **Unmodified.**
3. Hands the same two `ExperimentResult`s to
   `app.evaluation.statistics.Analyzer.compare_experiments` for
   significance/effect size. **Unmodified.**
4. Formats a one-sentence conclusion from the numbers steps 2-3 already
   computed — pure string templating, no computation.

Five of the seven ablation types (`no_exploration`, `alpha_sweep`,
`heuristic_only`, `linucb_only`, `alternative_reward_definitions`'s
*baseline* arm) need nothing beyond an `ExperimentConfig` with the right
`policy_name`/`alpha` — already fully supported. Two types need a
policy `ExperimentRunner` didn't already ship
(`random_critic_selection`, `reduced_context_features`); one needs a
reward *definition* it didn't already ship
(`alternative_reward_definitions`). `ExperimentRunner` (via its
`policy_factory` constructor argument) and `RewardCalculator` (via its
`strategy` constructor argument) were **already built for exactly this
kind of extension** — see their own docstrings. This module supplies
three small implementations of those existing extension points and
nothing else:

- **`RandomCriticPolicy`** — selects uniformly at random from the
  candidates. Deterministic (holds its own seeded `random.Random`,
  seeded from `ExperimentConfig.random_seed`). Returns
  `app.policy.models.PolicyDecision` — an existing model, not a new one.
- **`ReducedContextPolicy`** — wraps another `ReplayablePolicy`, keeping
  only the first `keep_fraction` of `context.feature_order` (via
  `ContextVector.model_copy`, never mutating the original) before
  delegating. `ContextVector`/`ReplayEngine` are untouched.
- **`QualityOnlyRewardStrategy`** — a `BaseRewardStrategy` subclass
  where reward is `aggregated_quality_score` alone (cost/latency/
  completion/correction ignored). `app/reward` is untouched.

## Supported `ablation_type`s

| `ablation_type` | Baseline arm | Candidate arm | Reads from `metadata` |
|---|---|---|---|
| `no_exploration` | `LinUCBPolicy(alpha=baseline_alpha, default 1.0)` | `LinUCBPolicy(alpha=0)` | `baseline_alpha` (optional) |
| `alpha_sweep` | `LinUCBPolicy(alpha=baseline_alpha, default 1.0)` | `LinUCBPolicy(alpha=metadata['alpha'])` | `alpha` (**required**), `baseline_alpha` (optional) |
| `random_critic_selection` | `baseline_policy` (e.g. `HeuristicPolicy`) | `RandomCriticPolicy` | `baseline_alpha` (optional, if baseline is LinUCB) |
| `heuristic_only` | `LinUCBPolicy(alpha=baseline_alpha, default 1.0)` | `HeuristicPolicy` | `baseline_alpha` (optional) |
| `linucb_only` | `HeuristicPolicy` | `LinUCBPolicy(alpha=candidate_alpha, default 1.0)` | `candidate_alpha` (optional) |
| `reduced_context_features` | `LinUCBPolicy(alpha)`, full context | Same policy, `ReducedContextPolicy`-wrapped | `alpha`, `keep_feature_fraction` (default `0.5`) |
| `alternative_reward_definitions` | Same policy, `WeightedRewardStrategy` | Same policy, `QualityOnlyRewardStrategy` | `alpha` (if policy is LinUCB) |

`baseline_policy`/`candidate_policy` on `AblationConfig` both label the
result (recorded in `AblationResult.metadata`) *and*, for most ablation
types, name which policy to actually build — see the table.

## Both arms share `random_seed`/`num_runs`

`AblationRunner.run`'s `num_runs`/`random_seed`/`candidate_actions`
arguments are passed **identically** to both arms' `ExperimentConfig`s.
Per `app/evaluation/statistics/README.md`, this means run `i` of the
baseline and run `i` of the candidate replay the *identical* bootstrap
resample — the comparison `Analyzer.compare_experiments` performs is
therefore genuinely paired, not just index-aligned.

## Usage

```python
runner = AblationRunner(repository=experience_repository)

# One ablation:
config = AblationConfig(
    experiment_name="my-study", baseline_policy="LinUCBPolicy",
    candidate_policy="LinUCBPolicy", ablation_type="no_exploration",
)
result = runner.run(config, num_runs=20, random_seed=42)

# All seven types (ten configs — alpha_sweep covers 0.25/0.5/1.0/2.0):
results = runner.run_all("my-study", num_runs=20, random_seed=42)

report = AblationReportGenerator()
report.to_markdown(results)   # summary table + ranking + best/worst + observations
report.to_csv(results)
report.to_json(results)
```

## Known interaction with earlier, already-documented limitations

- **`heuristic_only`, `linucb_only`, `no_exploration`, most of
  `alpha_sweep`, and `reduced_context_features` will typically report
  `test_used = "degenerate_zero_variance"` with `reward_difference =
  0.0`.** As documented in `app/evaluation/experiments/README.md` and
  `app/evaluation/offline/README.md`, an *untrained* `HeuristicPolicy`
  and an *untrained* `LinUCBPolicy` both deterministically select the
  same critic under the offline-replay `ContextVector` vocabulary
  (neither reads features this framework's offline context builder
  produces in a way that differentiates them) — so every paired run
  matches identically, and reducing the context or changing alpha
  changes nothing about that outcome either. This is not a flaw in the
  ablation framework; it faithfully reports that, **for these untrained
  policies specifically**, the ablated variants are indistinguishable
  from the baseline. `random_critic_selection` and
  `alternative_reward_definitions` are unaffected by this and reliably
  produce genuine variation (a random policy differs from any
  deterministic one by construction; a different reward *definition*
  changes the measured reward regardless of which critic was selected).
- To see `no_exploration`/`alpha_sweep`/`reduced_context_features`
  produce genuine variation, inject a **pre-trained** `LinUCBPolicy` via
  a custom `policy_factory` on a directly-constructed `ExperimentRunner`
  (see `app/evaluation/experiments/README.md`'s documented pattern) —
  out of scope for `AblationRunner` itself, which (per the Research
  Constraints) never trains a policy.

## Explicit non-goals

- No changes to `app/graph`, `app/router`, `app/policy_engine`,
  `ReplayEngine`, `ExperimentRunner`'s replay/aggregation logic,
  `Benchmark`, `Analyzer`, `app/reward`, `app/experience`, or
  `app/context`.
- No mutation of any `ExperienceRecord`/`ExperimentResult`/`ReplayResult`
  — every read is via plain attribute access on an already-frozen model.
- No PPO, no reinforcement learning, no live/online learning — no
  policy's `.update()` is ever called.
- No randomness outside `RandomCriticPolicy`'s own seeded
  `random.Random` (itself seeded from `ExperimentConfig.random_seed`):
  identical inputs always produce a bit-identical `AblationResult`.
