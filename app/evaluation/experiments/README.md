# Experiment Framework

Automates reproducible experiments over the existing
[Offline Replay Framework](../offline/README.md): run a policy `N`
independent times against stored `ExperienceRecord`s, aggregate the
statistics, and export the results.

> **Scope:** experiment orchestration only. No changes to `app/graph`,
> `app/router`, `app/policy_engine`, `app/experience`, `app/reward`,
> `app/context`, `app/evaluation/offline/replay.py`
> (`ReplayEngine`/`ReplayablePolicy`), or `app/evaluation/offline/benchmark.py`
> (`Benchmark`) — all used as-is. No PPO, no reinforcement learning, and
> no live/online learning: `ExperimentRunner` never calls `.update()` on
> any policy.

## Files

| File | Responsibility |
|---|---|
| `models.py` | `ExperimentConfig`, `ExperimentResult`, `ConfidenceInterval`, `StatisticalSummary`. |
| `runner.py` | `ExperimentRunner`, `BootstrapExperienceRepository` — runs experiments. |
| `analyzer.py` | `Analyzer` — mean/std/min/max/confidence-interval/trend statistics. |
| `exporter.py` | `Exporter` — JSON/CSV/Markdown serialization. |

## Why bootstrap resampling

`ReplayEngine.replay()` is a pure, deterministic function of a
repository's contents and a policy's (never-updated-during-replay)
state. Calling it twice on the same repository and policy always
produces the exact same `ReplayResult` — that's what makes offline
replay trustworthy, but it also means naively calling it `N` times in a
loop would produce `N` identical results: `std_reward = 0`,
`minimum = maximum = mean`, and a degenerate confidence interval.

So for `num_runs > 1`, `ExperimentRunner` draws `num_runs` independent
**bootstrap resamples** (with replacement) of the source repository's
stored experiences — seeded by `ExperimentConfig.random_seed` — and
replays each resample separately. This is the standard technique for
estimating the variance of an off-policy value estimate from a fixed
batch of logged data (bootstrap the log, re-evaluate on each bootstrap
sample, take statistics across the samples), and it requires no change
to `ReplayEngine`, `ExperienceRepository`, or any policy:

- `BootstrapExperienceRepository` is a **new** implementation of the
  existing `ExperienceRepository` abstract interface (exactly the
  extensibility point its own docstring describes), used only to hold a
  resample. Unlike `InMemoryExperienceRepository`, it permits duplicate
  `experience_id`s — required since a bootstrap resample legitimately
  draws the same record more than once.
- No `ExperienceRecord` is ever copied or mutated (they're frozen
  Pydantic models); a resample only holds new references to the same
  instances `repository.list()` already returned.
- `config.num_runs == 1` skips resampling entirely and replays the
  source data directly, exactly once.

`Analyzer.confidence_interval` correspondingly uses the **empirical
percentile method** (2.5th/97.5th percentile of the per-run values via
`numpy.percentile`), not a normal-distribution approximation — the
standard, assumption-free pairing with bootstrap-resampled values.

## `ExperimentConfig` / `ExperimentResult`

`ExperimentConfig` fully determines one reproducible run:
`experiment_name`, `policy_name` (`"HeuristicPolicy"` or
`"LinUCBPolicy"` by default), `alpha` (LinUCB only), `random_seed`,
`num_runs`, `candidate_actions`, `metadata`.

`ExperimentResult` holds the full list of per-run `ReplayResult`s
(`runs`, for complete traceability) plus cross-run aggregates:
`average_reward`, `std_reward`, `average_quality`, `average_latency`,
`average_iterations`, `match_rate`, `critic_selection_frequency`. The
reward distribution's confidence interval, minimum, and maximum are
recorded under `metadata["reward_confidence_interval_95"]`,
`metadata["reward_minimum"]`, `metadata["reward_maximum"]`.

## `ExperimentRunner`

```python
runner = ExperimentRunner(repository=experience_repository)  # + optional DI overrides

baseline = runner.run(ExperimentConfig(
    experiment_name="baseline", policy_name="HeuristicPolicy",
    random_seed=42, num_runs=20,
))

candidates = runner.run_sweep([
    ExperimentConfig(
        experiment_name=f"linucb-alpha-{alpha}", policy_name="LinUCBPolicy",
        alpha=alpha, random_seed=42, num_runs=20,
    )
    for alpha in (0.25, 0.5, 1.0, 2.0)
])
```

Every collaborator is injectable: `reward_calculator`, `evaluator`,
`analyzer`, and `policy_factory` (defaulting to
`_default_policy_factory`, which supports `"HeuristicPolicy"` and
`"LinUCBPolicy"` by name — inject a custom factory to support any other
policy without modifying this module). A **fresh** policy instance is
built per run via `policy_factory`; nothing is ever reused or mutated
across runs, and `.update()` is never called on any policy anywhere in
this framework.

## Known limitation: alpha sweeps need a pre-trained policy to matter

Since `ExperimentRunner` never trains a policy, `_default_policy_factory`
builds a **fresh, untrained** `LinUCBPolicy` for every run. An untrained
`LinUCBPolicy`'s arms all start identical (`A = regularization * I`,
`b = 0`), so for any given context every arm's `upper_confidence_bound`
is `alpha * sqrt(xᵀ A⁻¹ x)` — the **same** value for every arm, scaled
by the **same** `alpha`. Scaling every arm's score by the same constant
never changes which arm wins the tie. Empirically, an alpha sweep
(`0.25`, `0.5`, `1.0`, `2.0`) over freshly-constructed policies produces
**identical** `average_reward`/`critic_selection_frequency` across every
alpha value — this is mathematically correct, not a bug, and is
consistent with the Research Constraint that policies are never updated
during an experiment.

To run a sweep where `alpha` actually matters, train a `LinUCBPolicy`
separately first (via direct `.update()` calls, exactly as
`app/evaluation/offline/README.md` documents for evaluating a trained
policy), then inject a `policy_factory` that returns that **same**
already-trained instance for every run:

```python
trained_policy = LinUCBPolicy(alpha=0.5)
# ... train it via trained_policy.update(...) elsewhere ...
runner = ExperimentRunner(repository=repo, policy_factory=lambda config: trained_policy)
```

## `Analyzer`

`mean`, `std_dev` (sample, `ddof=1`), `minimum`, `maximum`,
`confidence_interval` (percentile method), and `summarize` (all of the
above in one `StatisticalSummary`) operate on any `Sequence[float]`.
`reward_trend`/`quality_trend`/`latency_trend` return each run's
respective `ReplayResult` field, in run order — the ordered sequence of
per-run values, useful for inspecting run-to-run spread. With an
untrained policy, this sequence reflects bootstrap-resampling variance
only, never learning progression (there is none).

## `Exporter`

`to_json`/`to_csv`/`to_markdown` return strings (`list[ExperimentResult] -> str`,
testable without touching the filesystem); `export(results, path)`
writes one of those formats to `path`, inferred from its suffix
(`.json`, `.csv`, `.md`/`.markdown`). `to_csv` emits one row per
`(experiment, run)` pair — the finest-grained view, for downstream
recomputation. `to_markdown` emits one row per experiment's *aggregate*
statistics — a human-readable summary table.

## Explicit non-goals

- No graph integration, no changes to `router_node`/`policy_engine_node`.
- No changes to `app/experience`, `app/reward`, `app/context`,
  `ReplayEngine`, `ReplayablePolicy`, `OfflineEvaluator`, or `Benchmark`.
- No live/online learning, no policy updates during an experiment.
- No PPO, no reinforcement learning.
- No randomness outside `ExperimentRunner`'s own seeded
  `random.Random(config.random_seed)` — replaying the same config
  against the same source data always produces a bit-identical
  `ExperimentResult`.
