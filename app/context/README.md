# Context Encoding Layer

This module converts an `AgentState` (and, optionally, a matching
`ExperienceRecord`) into a deterministic, numeric `ContextVector` — a
ready-made input representation for a future Contextual Bandit / Offline
RL / PPO / Q-learning policy, without requiring any change to this module
when that policy arrives.

> **Scope:** deterministic encoding and rescaling only. No reinforcement
> learning, no contextual bandits, no policy optimization, no learning of
> any kind. Every feature and every normalization bound is a **fixed
> constant chosen at implementation time** — nothing is fitted from a
> batch of data. Identical inputs always produce identical
> `ContextVector`s.
>
> As of the Policy Abstraction Layer task, `ContextEncoder` is used by
> `policy_engine_node` (see `app/graph/nodes.py`) to build the
> `ContextVector` passed to `BasePolicy.select_action` (see `app/policy`).
> `ContextNormalizer` remains unused by the graph so far, available for a
> future policy that wants normalized input.

## Files

| File | Responsibility |
|---|---|
| `models.py` | `ContextVector` — the frozen, self-describing output. |
| `encoder.py` | `ContextEncoder` — `(AgentState, ExperienceRecord?) -> ContextVector`. |
| `normalizer.py` | `ContextNormalizer` — rescales a `ContextVector`'s features into `[0.0, 1.0]`. |

## `ContextVector`

A frozen (`extra="allow", frozen=True`) Pydantic v2 model, the same
"immutable historical record" pattern used by `ExperienceRecord`,
`RewardSignal`, and `ExecutionMetrics`: `context_id`,
`source_execution_id`, `features`, `feature_order`, `normalized`,
`normalization_strategy`, `timestamp`, `metadata`.

`feature_order` exists because a future policy will want to build a
fixed-width numeric array (e.g. for a `numpy`/`torch` tensor) — relying
on `dict` iteration order alone is fragile and undocumented; this makes
the canonical ordering explicit and stable.

## `ContextEncoder`

### Why `state` alone drives `features`, and `experience` is kept separate

A contextual bandit's "context" is, by definition, the observation
available **before** an action is chosen and **before** an outcome is
known. `ContextEncoder.encode(state, experience=None)` therefore encodes
every entry in `features` from `AgentState` alone — the pre-decision
observation a real policy would actually have. If an `ExperienceRecord`
is also passed (useful for offline/retrospective analysis, where the
outcome is already known), its `latency` and `estimated_cost` are encoded
*separately*, under `metadata["experience_derived"]`, specifically so
they can never be mistaken for information available at decision time.
Mixing them into `features` would silently leak future information into
what is supposed to be a pre-decision context.

### Encoded features

27 named features, each either copied, a fixed categorical→numeric
mapping, or a simple fixed-formula ratio/aggregate:

| Feature | Encoding |
|---|---|
| `iteration_count`, `max_iterations` | direct copy |
| `iteration_ratio` | `iteration_count / max_iterations`, clamped to `[0, 1]` |
| `error_feature_count`, `worker_output_count`, `critic_score_count`, `selected_critics_count`, `retrieved_memories_count`, `correction_history_count` | `len(...)` of the corresponding `AgentState` list/dict |
| `aggregated_quality_score` | direct copy, imputed to `0.0` if `None` |
| `has_aggregated_quality_score` | `1.0`/`0.0` missingness flag |
| `safety_status_code` | fixed ordinal map (`SAFETY_STATUS_CODES`): unknown=0, safe=1, flagged=2, blocked=3 |
| `execution_status_code` | fixed ordinal map (`EXECUTION_STATUS_CODES`): pending=0 … cancelled=5 |
| `is_code_task` | `1.0` if the resolved task type is `"code"`, else `0.0` |
| `has_task_type` | `1.0`/`0.0` missingness flag |
| `average_critic_score`, `max_critic_score`, `min_critic_score` | aggregate of `critic_scores.values()`, `0.0` if empty |
| `uncertainty`, `risk`, `task_complexity`, `memory_relevance`, `requires_self_correction`, `requires_meta_critic`, `is_code_output`, `iteration_pressure`, `attempt_pressure` | mirror `app/policy_engine/scorer.py`'s `HeuristicPolicyScorer.extract_features` exactly — see below |

An unrecognized status string (e.g. a future `ExecutionStatus` value) is
encoded as `UNRECOGNIZED_STATUS_CODE = -1.0` rather than raising —
graceful degradation, consistent with every other module in this
project.

Task type resolution mirrors the same `state.task_type` →
`state.planner_output.task_type` fallback used elsewhere in ACRF, but is
duplicated locally rather than imported, since `app/context` must not
depend on `app/graph`.

### The nine `HeuristicPolicyScorer`-parity features

`BasePolicy.select_action` (see `app/policy`) receives only a
`ContextVector`, not the original `AgentState`. `HeuristicPolicy` — the
policy that replaces the old, directly-`AgentState`-based
`HeuristicPolicyScorer.score()` call — therefore needs everything that
scorer used to read (`error_features`, `memory_context`,
`planner_output`, `iteration_count`, `worker_outputs`) to already be
present as plain numeric `ContextVector` features. `uncertainty`, `risk`,
`task_complexity`, `memory_relevance`, `requires_self_correction`,
`requires_meta_critic`, `is_code_output`, `iteration_pressure`, and
`attempt_pressure` were added here for exactly that reason, computed with
formulas that **exactly mirror**
`app/policy_engine/scorer.py`'s private extraction functions (same
thresholds, same clamping) — verified in
`tests/test_context_encoder.py::test_heuristic_scorer_parity_features_match_extract_features_exactly`,
which asserts equality against `HeuristicPolicyScorer.extract_features`
directly.

The extraction logic is **duplicated, not imported**, so `app/context`
still has no dependency on `app/policy_engine` — the same "small modules
stay independent" convention `app/correction_policy` already established
for exactly this kind of small, well-contained overlap. This is a
deliberate trade-off: a small maintenance cost (two copies of the same
formulas) in exchange for `app/context` remaining a leaf module with no
dependency on the policy layer built on top of it.

Note `is_code_output` (derived from the *worker output text*, via the
latest error feature's `output_type`) is deliberately distinct from the
existing `is_code_task` (derived from *task type*) — they usually
correlate but are independent signals that can disagree, and both remain
available.

### `context_id`

`sha256(f"context|{session_id}|{task_id}|{iterations}")` — deterministic
and collision-resistant, salted with a fixed `"context|"` prefix so it
never collides with `ExperienceRecord.experience_id` /
`ExecutionMetrics.execution_id`, even though all three are derived from
the same `(session_id, task_id, iterations)` triple.

## `ContextNormalizer`

Fixed-bounds min-max scaling: `FEATURE_BOUNDS` is a hard-coded
`{feature_name: (min, max)}` table (one entry per feature `ContextEncoder`
produces), injected via the constructor (dependency injection) and
defaulting to that table. `normalize(context)` returns a **new**
`ContextVector` (the input is frozen and never mutated) with every value
clamped to `[0.0, 1.0]` and `normalized=True`.

**Bounds are fixed constants, never fitted from data.** Computing
min/max from a batch of observed `ContextVector`s would be learning that
distribution's statistics — exactly what this module must not do. A
feature name absent from the bounds table (e.g. a future encoder
version's new feature) passes through unchanged rather than raising.

## Future compatibility

`ContextEncoder` and `ContextNormalizer` depend only on the plain
`ContextVector` model and on `AgentState`/`ExperienceRecord` — never on
any concrete learning algorithm. A future Contextual Bandit, PPO, or
Q-learning implementation consumes `ContextVector.features` (or the
normalized version) as-is; none of it requires any change to this module.

## Explicit non-goals

- No reinforcement learning, contextual bandits, or policy optimization.
- No fitting of normalization bounds from data.
- No router, policy, critic, reward, experience, or metrics changes.
- No graph topology changes — this module is not wired into any node.
- No LLM calls.
- No randomness — every run of the same input is bit-for-bit identical.
