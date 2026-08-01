# Reward Engine

This module converts a completed `ExperienceRecord` (see `app/experience`)
into a deterministic `RewardSignal`, so future adaptive-learning
algorithms — contextual bandits, offline RL, PPO, Q-learning — have a
stable, pre-computed reward to consume without needing to touch this
module.

> **Scope:** deterministic heuristics only. No reinforcement learning, no
> contextual bandits, no PPO, no DQN, no Q-learning, no neural networks,
> no replay buffers, no policy optimization, no LLM calls. Every weight
> in `strategy.py` is a fixed constant chosen at implementation time —
> there is no training, no gradient updates, and no randomness anywhere
> in this module. Identical `ExperienceRecord` values always produce
> identical `RewardSignal`s.

## Files

| File | Responsibility |
|---|---|
| `models.py` | `RewardSignal` — the frozen, self-describing output. |
| `strategy.py` | `BaseRewardStrategy` (abstract) and `WeightedRewardStrategy` (the one concrete, fixed-weight implementation) — the actual component functions live here. |
| `calculator.py` | `RewardCalculator` — the thin, dependency-injected entry point: `ExperienceRecord -> RewardSignal`. No repository access, no learning, no policy updates. |

## `RewardSignal`

A frozen (`extra="allow", frozen=True`) Pydantic v2 model, matching the
"immutable historical record" pattern already used by `ExperienceRecord`:
`reward`, `quality_reward`, `efficiency_penalty`, `cost_penalty`,
`latency_penalty`, `correction_penalty`, `completion_bonus`,
`confidence`, `strategy`, `explanation`, `metadata`.

## `WeightedRewardStrategy`

### Components

Five independent, bounded components, each a pure function of one
`ExperienceRecord` field:

| Component | Derived from | Formula | Bound |
|---|---|---|---|
| `quality_reward` | `aggregated_quality_score` | `clamp(score, 0, 1) * 1.0` | `[0.0, 1.0]` |
| `correction_penalty` | `iterations` | `iterations * 0.1` | `[0.0, 0.5]` |
| `cost_penalty` | `estimated_cost` | `max(0, cost) * 0.01` | `[0.0, 0.5]` |
| `latency_penalty` | `latency` (seconds) | `max(0, latency) * 0.05` | `[0.0, 0.5]` |
| `completion_bonus` | `execution_status` | `+0.2` if `"completed"`, `-0.3` if `"failed"`, else `0.0` | `[-0.3, 0.2]` |

`efficiency_penalty` is a **rollup**, not an independent component:
`efficiency_penalty = cost_penalty + latency_penalty`, reported for
consumers that want one combined "resource efficiency" signal. It is
**not** subtracted again in `reward` — its two parts already are,
individually — to avoid double-counting.

### Total reward

```
reward = quality_reward + completion_bonus
         - cost_penalty - latency_penalty - correction_penalty
```

### "Failed execution → negative adjustment"

There is no separate `failure_penalty` field in the spec's `RewardSignal`
fields, so this requirement is implemented as `completion_bonus` being
**bidirectional**: positive (`+0.2`) for `"completed"`, negative
(`-0.3`) for `"failed"`, and neutral (`0.0`) for any other status
(`"pending"`, `"running"`, or an unrecognized future value). This is
documented explicitly here and in `strategy.py` since the field name
alone doesn't make the sign convention obvious.

### `confidence`

The fraction (`0.0`-`1.0`) of three optional signals —
`aggregated_quality_score`, `estimated_cost`, `latency` — that were
actually present (non-`None`) on the source `ExperienceRecord`. A
`RewardSignal` computed from an experience missing all three optional
fields still has a fully valid `reward` (every missing signal degrades to
a neutral `0.0` contribution), but `confidence == 0.0` flags that the
computation had little to go on.

### Graceful degradation

Every extractor function treats a missing/`None` field as a neutral
(`0.0`) contribution rather than raising, and clamps out-of-range values
(e.g. a quality score outside `[0, 1]`, a negative cost) rather than
letting them distort the total unboundedly. An `ExperienceRecord` with no
optional fields set at all still produces a well-formed `RewardSignal`
(`reward = completion_bonus` only, `confidence = 0.0`).

## `RewardCalculator`

```python
class RewardCalculator:
    def __init__(self, strategy: BaseRewardStrategy | None = None) -> None: ...
    def calculate(self, experience: ExperienceRecord) -> RewardSignal: ...
```

`strategy` is injected via the constructor (dependency injection),
defaulting to `WeightedRewardStrategy`. `RewardCalculator` itself contains
no computation — it only delegates — so a future
`ContextualBanditRewardStrategy`, `PPORewardStrategy`, or
`QLearningRewardStrategy` could be substituted with **zero changes to
`RewardCalculator`**, exactly the future-compatibility requirement this
module was designed around. `RewardCalculator` never touches a
repository and never updates any policy — it is a pure
`ExperienceRecord -> RewardSignal` function.

## Integration with `evaluation_node`

`app/graph/nodes.py::evaluation_node` — and only this node — now, right
after building the `ExperienceRecord` (via `ExperienceRecorder`,
unchanged from the previous task) and before storing it:

1. Computes `reward = RewardCalculator().calculate(experience)`.
2. Builds an **enriched** copy of the experience with the reward attached
   into its `metadata`:
   `experience.model_copy(update={"metadata": {**experience.metadata, "reward": reward.model_dump(mode="json")}})`
   — a copy, not a mutation, since `ExperienceRecord` is frozen.
3. Stores the **enriched** record into `DEFAULT_EXPERIENCE_REPOSITORY`
   (not the reward-less one — `ExperienceRecorder` is called without a
   repository this time specifically so nothing is stored before the
   reward is attached).
4. Writes both `state.memory_context["experience"]` (the enriched dump)
   and `state.memory_context["reward"]` (the `RewardSignal` dump).

No other node, and no graph topology, routing, policy, or critic code,
is touched.

## Explicit non-goals

- No reinforcement learning, contextual bandits, PPO, DQN, or Q-learning.
- No neural networks, replay buffers, or policy optimization.
- No LLM calls.
- No repository access inside `RewardCalculator` or `strategy.py`.
- No change to graph topology, routing, policy, or critic behavior.
- No randomness — every run of the same input is bit-for-bit identical.
