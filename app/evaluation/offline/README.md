# Offline Replay & Benchmark Framework

Replays already-stored `ExperienceRecord`s against a policy, and compares
policies fairly — entirely offline. No LLM call, no graph execution, and
no new `ExperienceRecord` is ever created.

> **Scope:** offline replay and benchmarking only. No graph integration,
> no changes to `router_node`, `policy_engine_node`, `app/experience`,
> `app/reward`, `app/context`, or any `app/policy*` module, no PPO, and
> no reinforcement learning. `ReplayEngine.replay` — the original,
> deterministic mode, unchanged — never calls an `update` method on the
> policy it replays, so replaying a policy through it can never train
> it. `ReplayEngine.replay_with_learning` is an additional, explicitly
> opt-in mode (see below) that *does* call `update`, sequentially, so
> the two modes can be compared side by side; it does not alter
> `replay`'s behavior in any way.

## Files

| File | Responsibility |
|---|---|
| `models.py` | `ReplayStep`, `ReplayResult`, `BenchmarkResult` — the structured, immutable outputs. |
| `replay.py` | `ReplayEngine`, `ReplayablePolicy`, `TrainablePolicy`, `build_offline_context_vector` — the replay mechanics. |
| `evaluator.py` | `OfflineEvaluator` — aggregates one policy's replay into a `ReplayResult`. |
| `benchmark.py` | `Benchmark` — compares two `ReplayResult`s into a `BenchmarkResult`. |

## The replay method

`ReplayEngine` implements the **replay method** for offline (off-policy)
evaluation of logged bandit feedback (Li et al., 2011, "Unbiased Offline
Evaluation of Contextual-bandit-based News Article Recommendation
Algorithms"). For every stored experience:

1. Build a `ContextVector` from the experience alone (`build_offline_context_vector`).
2. Ask the policy being replayed: `policy.select_action(context, candidate_actions)`.
3. If the policy's selection exactly matches (as a set) the critic(s)
   that were actually recorded for that experience, the experience's
   already-recorded reward is *replayed* for that matched decision
   (`RewardCalculator.calculate(experience)`), never fabricated or
   estimated.
4. If the policy would have chosen differently, the experience is
   **skipped** for this policy's evaluation. Its outcome under an action
   that was never actually taken is unknown, and estimating it would
   bias the result.

This is why two different policies replayed against the *same*
repository generally produce different `total_experiences` counts:
each policy only "counts" the subset of history it would have
reproduced.

> **What "replay" does and does not claim.** Two distinct properties are
> easy to conflate, so they are stated separately here.
>
> 1. **Reward-attribution correctness for the matched subset.** Whenever
>    the policy's action exactly matches the logged action, replaying
>    that experience's already-recorded reward is exact, not estimated.
>    This holds unconditionally.
> 2. **Estimator unbiasedness, in the formal sense Li et al. prove.**
>    Their proof additionally assumes the *logging* policy selected
>    actions with known, non-degenerate probability (uniform random, in
>    their deployment) — so that every candidate action had some chance
>    of being logged for a given context, and matches can be reweighted
>    (or, in the simplified uniform case, counted directly) into an
>    unbiased estimate of a candidate policy's performance across the
>    *whole* action space.
>
>    ACRF's logged actions come from `router_node`'s fixed, deterministic,
>    task-type-based rule — not a randomized exploration policy — so
>    property (2) does not hold here, and this module does not claim it:
>    no inverse-propensity weighting is computed. A `ReplayEngine` result
>    should be read as an exact replay of matched historical decisions,
>    which is useful for comparing candidate policies against what
>    actually happened, but **not** as a statistically unbiased estimate
>    of a candidate policy's performance had it been deployed live across
>    the full action space. This distinction does not require any code
>    change (`replay`'s algorithm was already exactly what is described
>    above); it is a correction to how the result should be described.

## `build_offline_context_vector`

`app.context.ContextEncoder` cannot be reused here — it only encodes a
live `AgentState`, and a stored `ExperienceRecord` is not one (offline
replay, by definition, has no live `AgentState`). `build_offline_context_vector`
is therefore a separate, self-contained function that builds a
`ContextVector` from `ExperienceRecord` fields alone.

`ExperienceRecord` is built exactly once, at the *end* of an episode
(`evaluation_node`, via `ExperienceRecorder`) — there is no separate,
pre-decision snapshot in the current data model. A policy's input must
not include information that would not genuinely be available before a
live routing decision, so this function reads only the subset of fields
verified invariant across a whole recorded episode:

| Feature | Source | Why it is safe |
|---|---|---|
| `is_code_task` | `state_features["task_type"] == "code"` | `AgentState.task_type` is set once, before the graph runs, and never reassigned by any node. |
| `has_task_type` | `state_features["task_type"]` is present | Same as above. |
| `plan_complexity` | `len(state_features["planner_output"]["decomposition"]) / 5`, clamped to `[0, 1]` | `AgentState.planner_output` is assigned exactly once, by `planner_node` (the graph's first node); `self_correction_node` does not re-invoke the planner. Mirrors `app/context/encoder.py`'s `_extract_plan_complexity`. |
| `max_iterations` | `state_features["max_iterations"]`, min-max scaled against `[1, 20]` | A fixed configuration/budget value, never reassigned by any node. Mirrors `app/context/normalizer.py`'s `FEATURE_BOUNDS["max_iterations"]`. |

A prior version of this function instead read `aggregated_quality_score`,
`iterations`, `latency`, `estimated_cost`, critic-score count/average,
selected-critic count, correction-decision presence, and (from
`state_features`) `error_feature_count`/`worker_output_count`. Every one
of those fields reflects the episode's *outcome*, the action *actually
taken*, or a count that can change across an episode's later iterations
(`error_feature_extractor_node` re-runs on every `worker ->
error_feature_extractor` pass) — using any of them as a policy *input*
lets the policy's decision be informed by information that would not
exist yet at decision time (target/outcome leakage). This was a defect,
not a design choice, and has been corrected; the function's signature and
return type (`ContextVector`) are unchanged, but the *content* of
`.features` is a deliberate, breaking change from the prior version. Any
code or notebook relying on the old feature names by string must be
updated; `context.feature_order` should always be preferred over
hardcoding feature names.

`build_offline_context_vector` does not modify, import internals from, or
duplicate `app/context/encoder.py` — it is independent by design, the
same "small modules stay independent" convention `app/correction_policy`
and `app/context` already follow.

> **Known limitation:** `HeuristicPolicy` scores critics from nine
> specifically-named features (`uncertainty`, `risk`, `task_complexity`,
> `memory_relevance`, `requires_self_correction`, `requires_meta_critic`,
> `is_code_output`, `iteration_pressure`, `attempt_pressure` — see
> `app/policy/heuristic_policy.py`). None of those names are produced by
> `build_offline_context_vector` (before or after the correction above),
> so under replay they all default to a neutral `0.0`/`False`, and
> `HeuristicPolicy` selects the **same** critic for every experience
> (empirically, `CodeCritic`, since ties are broken alphabetically). This
> is expected, not a bug: those signals live on
> `AgentState.error_features`/`planner_output`, which are not preserved
> verbatim on `ExperienceRecord`, and reconstructing them would mean
> either changing `app/context` (out of scope) or deep, fragile parsing
> of `state_features["error_features"]`'s raw dicts (a third duplicate of
> the same weight tables, for uncertain benefit). `HeuristicPolicy`
> replay results are therefore mechanically correct — the "replay
> method" itself is exercised faithfully — but not a meaningful proxy for
> `HeuristicPolicy`'s live behavior. The leakage correction above does
> not change this: it was already true, for an unrelated reason (a naming
> mismatch, not a leakage issue), and remains equally true now that the
> context is smaller.
>
> Policies without a fixed feature vocabulary (`LinUCBPolicy`, and any
> future contextual bandit / offline RL policy) are **not** affected:
> they read `context.features` positionally via `context.feature_order`,
> so they respond to whatever this function produces, regardless of
> naming. With only four features remaining, and many logged episodes
> likely sharing the same `task_type`/`max_iterations`, expect `LinUCBPolicy`
> to see less-differentiated context under offline replay than the prior,
> leakier version provided — an intentional, necessary consequence of
> removing leakage, not a new defect. See "Remaining risks" in the P0
> review report for the deeper, out-of-scope fix (extending
> `ExperienceRecord` with a genuine pre-decision snapshot).

## `ReplayablePolicy`

```python
class ReplayablePolicy(Protocol):
    def select_action(self, context: ContextVector, candidate_actions: list[str]) -> object: ...
```

A structural (not nominal) interface. `HeuristicPolicy`
(`select_action(context, candidate_critics) -> PolicyDecision`) and
`LinUCBPolicy` (`select_action(context, actions) -> LinUCBSelection`)
both satisfy it without declaring so, and without this module importing
either class. `_extract_selected_critics` then normalizes whatever comes
back by duck-typing: a `selected_critics: list[str]` attribute (as on
`PolicyDecision`) or a `selected_action: str` attribute (as on
`LinUCBSelection`) — any future policy decision object need only expose
one of these two names to be replayable, with zero changes to this
module.

## `ReplayEngine`

```python
engine = ReplayEngine(
    repository=experience_repository,
    policy=heuristic_policy,          # or a pre-trained LinUCBPolicy, etc.
    reward_calculator=RewardCalculator(),
)
steps = engine.replay()  # list[ReplayStep], one per matched experience
```

`repository` is read via `.list()` only — never written to.
`experience` (a frozen Pydantic model) is never mutated. No `update` is
ever called on `policy`: **evaluating** a policy via replay can never
**train** it. To evaluate a policy that benefits from prior training
(e.g. `LinUCBPolicy`), train it separately (calling `.update()`
directly, outside this framework) before constructing the `ReplayEngine`
— training is out of scope here by design.

`candidate_actions` defaults to `DEFAULT_CANDIDATE_CRITICS`
(`LogicCritic`, `CodeCritic`, `FactCritic`, `MetaCritic`), mirroring
`app/graph/nodes.py`'s live candidate set — duplicated, not imported, so
this module has no dependency on `app/graph`.

### `replay_with_learning`: the optional Sequential Replay Learning Mode

```python
trainable_policy = LinUCBPolicy(alpha=0.5)
engine = ReplayEngine(repository=experience_repository, policy=trainable_policy, reward_calculator=RewardCalculator())

training_steps = engine.replay_with_learning()  # list[ReplayStep] -- same shape replay() returns
```

An additional method on the same `ReplayEngine`, added without changing
one line of `replay`'s own implementation (so the two are always safe
to compare). For every stored experience, in `repository.list()`'s
order:

1. Build the context and call `policy.select_action` — same as `replay`.
2. If the selection doesn't match the experience's recorded critic(s),
   skip it — same matching rule as `replay`; this mode still never
   trains on a reward for an action that was never actually taken.
3. Otherwise, compute the reward (`reward_calculator.calculate`) and
   **immediately call `policy.update(context, critic, reward)`** for
   each selected critic, *before* moving to the next experience.

Because the policy's internal state changes mid-pass, order matters and
the policy genuinely *learns* across the call — unlike `replay`, where
every experience is scored against the exact same, unchanging policy
state. Requires a policy implementing `TrainablePolicy`
(`select_action` + `update(context, action, reward)`, the exact shape
`LinUCBPolicy` already has); calling it with a non-trainable policy
(e.g. `HeuristicPolicy`) raises `AttributeError` immediately, before any
replay happens.

To compare the two modes, construct two separate, identically-seeded
policy instances (one for each engine) and call `replay()` on one,
`replay_with_learning()` on the other — `replay()`'s policy is never
mutated, so a single shared instance would otherwise let the two calls
interfere with each other.

## `OfflineEvaluator`

```python
result = OfflineEvaluator().evaluate(engine, policy_name="HeuristicPolicy")  # -> ReplayResult
```

Stateless; aggregates `engine.replay()`'s matched `ReplayStep`s into a
`ReplayResult`. A missing `quality`/`latency` on a matched step
contributes `0.0` to its average (matching `app/reward`'s established
convention for missing optional signals). With zero matches, every
numeric field is `0.0` and `critic_selection_frequency` is empty — no
division by zero.

## `Benchmark`

```python
benchmark_result = Benchmark().compare(baseline=heuristic_result, candidate=linucb_result)
```

Pure computation over two already-computed `ReplayResult`s.
`reward_improvement`/`quality_improvement`/`latency_difference`/
`iteration_difference` are all `candidate - baseline`. `winner` is
whichever `policy_name` had the strictly higher `average_reward`, or
`"tie"` if equal — `average_reward` is this framework's single source of
truth for "which policy performed better," matching reward's existing
role elsewhere in ACRF.

## Explicit non-goals

- No graph integration: not imported by `app/graph/nodes.py`.
- No changes to `router_node`, `policy_engine_node`, `app/experience`,
  `app/reward`, `app/context`, or any `app/policy*` module.
- No live learning, no online updates during `replay()` — that method
  never calls `policy.update(...)`, unchanged from its original form.
  `replay_with_learning()` is the sole, explicitly opt-in exception,
  added alongside it without modifying it.
- No PPO, no reinforcement learning, no new experiences.
- No randomness — replaying the same repository against the same
  (already-fixed) policy state always produces identical `ReplayStep`s;
  `replay_with_learning()` is equally deterministic (same repository +
  same starting policy state + same reward calculator always produce
  the same sequence of updates and the same resulting trained state).
