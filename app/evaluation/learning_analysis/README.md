# Learning Analysis

Extends the existing experiment analysis with **learning-curve
metrics**, computed purely by reading an already-completed sequential
replay — no new infrastructure, no architecture changes, no changes to
`ReplayEngine`, and no changes to any policy.

> **Scope:** read-only analysis only. This module never calls
> `select_action`/`update` on a policy, never reads an
> `ExperienceRepository`, never constructs a `ReplayEngine`, and never
> creates or stores an `ExperienceRecord`. It only ever transforms an
> already-produced `list[app.evaluation.offline.ReplayStep]` — the
> return value of a completed
> `ReplayEngine.replay()`/`replay_with_learning()` call — into derived
> metrics. No PPO, no reinforcement learning, no policy updates.

## Files

| File | Responsibility |
|---|---|
| `models.py` | `LearningCurve` — the structured, immutable result. |
| `analyzer.py` | `LearningAnalyzer` — computes every metric. |
| `report.py` | `LearningReportGenerator` — Markdown/CSV/JSON output. |

## Input: a completed sequential replay

```python
steps = engine.replay_with_learning()   # list[ReplayStep], see app/evaluation/offline
curve = LearningAnalyzer().analyze(steps)
```

`steps` is most meaningful when it comes from
`ReplayEngine.replay_with_learning()` (see
`app/evaluation/offline/README.md`): its order reflects a policy
learning, sequentially, from the replayed log, which is what makes a
"learning curve" meaningful. `LearningAnalyzer` does not require this,
though — it is a pure function of *any* ordered `list[ReplayStep]`, so
analyzing a plain `replay()` result is equally valid (and a useful
baseline: a non-learning policy's curve is flat, with a
`learning_rate_estimate` near `0.0` and `convergence_point == 0`).

## What each field means

- **`reward_per_step`** — each step's `reward`, unchanged, in order.
- **`cumulative_reward`** — the running sum.
- **`instantaneous_regret`** / **`cumulative_regret`** — see below.
- **`average_reward`** — the single overall mean (a summary statistic;
  contrast with `moving_average_reward`, a series).
- **`moving_average_reward`** — a trailing moving average (default
  window `10`, configurable), smoothing out step-to-step noise.
- **`metadata`** — `num_steps`, `convergence_point`,
  `learning_rate_estimate`, `moving_average_window`,
  `convergence_tolerance`, `best_reward_observed`, `worst_reward_observed`.
  `convergence_point` and `learning_rate_estimate` are not top-level
  `LearningCurve` fields (the spec for this model names only the six
  series/summary fields above); they are recorded here, the same
  convention every other `metadata` dict in this evaluation suite
  already follows for computed diagnostics that aren't in a model's
  fixed field list.

## Regret: a documented, necessary approximation

True bandit regret is `reward of the true optimal action - reward of
the action taken`, at every step. Offline replay never observes the
reward of an action it didn't take (that is exactly what makes the
"replay method" *unbiased* — see `app/evaluation/offline/README.md`),
so the true optimal is fundamentally unknowable from replay data alone.
`instantaneous_regret[i] = max(reward_per_step) - reward_per_step[i]`
— the best reward **actually observed anywhere in this completed
run** — is this module's deterministic, fully-computable substitute.

This is a **hindsight** (retrospective) regret, appropriate for
post-hoc analysis of a *completed* result (this module's whole premise,
per its Research Constraints) — not a **causal/online** definition using
only `reward_per_step[0..i]` (regret against the best seen *so far*,
which would be smaller and monotonically non-increasing as a reference
point). Both are legitimate, standard practical regret proxies used
when the true optimum is unavailable; this module uses the hindsight
definition because it answers "how far below the best outcome in this
log was each step," a natural question once the whole run is already
known. `instantaneous_regret` is always `>= 0`, and `cumulative_regret`
is therefore always monotonically non-decreasing.

## Convergence point

The earliest step index from which `moving_average_reward` never again
leaves a `+/- tolerance * range(moving_average_reward)` band around its
own final value (default `tolerance = 0.05`, i.e. 5% of the observed
range). Always a valid index for a non-empty curve (the last index
trivially qualifies); `None` only for an empty input.

## Learning rate estimate

The ordinary-least-squares slope of `reward_per_step` against the step
index — "reward change per step," a simple, deterministic linear-trend
summary. Positive: reward tended to rise over the run. Negative: it
tended to fall. Near zero: no clear linear trend (e.g. a flat,
non-learning `replay()` curve, or a curve that converged near the very
start and stayed flat thereafter).

## Worked example

Replaying 8 strongly-negative-reward `CodeCritic` experiences followed
by 12 strongly-positive-reward `FactCritic` experiences through
`replay_with_learning()` with an untrained `LinUCBPolicy` (see
`tests/test_offline_policy_replay.py`'s flip demonstration) produces
exactly the pattern you'd expect: `reward_per_step` jumps from `-0.325`
(the one matched, failed `CodeCritic` experience) to `1.175` for every
subsequent step (the policy switched to `FactCritic` after that first
update and stayed there); `instantaneous_regret` is `1.5` at step 0 and
`0.0` thereafter; `cumulative_regret` rises once, to `1.5`, then stays
flat; `moving_average_reward` climbs steadily toward `1.175` as the
window fills with good rewards; `convergence_point` lands at the step
the moving average first settles within its tolerance band; and
`learning_rate_estimate` is positive, correctly summarizing the overall
upward trend.

## `LearningReportGenerator`

```python
report = LearningReportGenerator()
report.to_markdown(curve)   # summary + a compact, evenly-sampled steps table (never a full dump)
report.to_csv(curve)         # one row per step: step, reward, cumulative_reward,
                              # instantaneous_regret, cumulative_regret, moving_average_reward
report.to_json(curve)        # the full LearningCurve, every step
```

`to_csv`/`to_json` are the full-detail exports — exactly the data
needed to plot each of the five supported figures (Reward Curve,
Cumulative Reward, Instantaneous Regret, Cumulative Regret, Moving
Average Reward: all five are columns in `to_csv`'s output). No plotting
library is used or required; this module supports **exporting the
data** for those figures, not rendering images.

## Explicit non-goals

- No new infrastructure, no architecture changes.
- No changes to `ReplayEngine` or to any policy — this module imports
  `ReplayStep` (a plain data model) from `app.evaluation.offline` and
  nothing else from it.
- No policy updates, no replay modifications — `LearningAnalyzer` never
  calls `.update()` or `.select_action()` on anything.
- No PPO, no reinforcement learning.
- No randomness — identical `steps` input always produces a bit-identical `LearningCurve`.
