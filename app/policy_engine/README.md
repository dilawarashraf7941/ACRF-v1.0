# Heuristic Policy Engine

This module replaces the constant placeholder policy score
(`PlaceholderPolicyEngine`, always `0.0` — see `app/policies/engine.py`)
with a real, deterministic, **feature-based heuristic** scoring, ranking,
and selection layer for candidate critics.

> **Scope:** deterministic heuristics only. No reinforcement learning, no
> Q-learning, no PPO, no DQN, no neural networks, no LLM calls, no API
> calls. Every weight in `scorer.py` is a fixed constant chosen at
> implementation time — there is no training, no gradient updates, and no
> randomness anywhere in this module. Identical `AgentState` values always
> produce identical scores; different values genuinely produce different
> scores.

## Files

| File | Responsibility |
|---|---|
| `scorer.py` | `HeuristicPolicyScorer` — extracts deterministic features from `AgentState` and computes one score per candidate critic. |
| `ranking.py` | `CriticRanking` — deterministically orders a `dict[str, float]` of scores, highest first, with alphabetical tie-breaking. |
| `selector.py` | `CriticSelector` — selects critics from a `CriticRanking` via top-1, top-k, or threshold strategies. |

## `HeuristicPolicyScorer`

### Inputs

Reads only five `AgentState` fields, exactly as specified:

- `error_features` — the latest entry's `metadata` (as written by
  `error_feature_extractor_node`), including the nested
  `metadata["profile"]` dump of the full `ErrorFeatureProfile`.
- `memory_context` — an optional `memory_context["memory_relevance"]`
  float, a forward-compatible hook for a future memory subsystem.
- `iteration_count`
- `planner_output` — specifically `planner_output.decomposition`'s length.
- `worker_outputs` — specifically the count of *extra* attempts beyond
  the first.

Notably, **`state.task_type` is never read** — it is not in the
specified input set, so scores are identical regardless of it. (This is
independent of `router_node`'s own task-type-based critic selection,
which is unaffected by this module.)

### Feature extraction

`HeuristicPolicyScorer.extract_features` turns those five fields into a
fixed `StateFeatures` record, each field normalized to `[0.0, 1.0]` (or a
plain `bool`):

| Feature | Derived from | Notes |
|---|---|---|
| `uncertainty` | `1 - confidence` | confidence from `error_features[-1].metadata["confidence"]`, falling back to `metadata["profile"]["confidence_score"]`, defaulting to full confidence (`0.0` uncertainty) if absent |
| `risk` | `risk_level` string → `_RISK_LEVEL_SCORES` | `low`/`medium`/`high`/`critical` → `0.0`/`0.4`/`0.75`/`1.0` |
| `task_complexity` | `max(` profile's `task_complexity` string, plan decomposition length / 5 `)` | combines both available signals |
| `memory_relevance` | `max(` profile's `memory_relevance`, `memory_context["memory_relevance"]` `)` | combines both available sources |
| `requires_self_correction` | profile flag | `bool`, defaults `False` |
| `requires_meta_critic` | profile flag | `bool`, defaults `False` |
| `is_code_output` | `metadata["output_type"] == "code"` | `bool` |
| `iteration_pressure` | `iteration_count / 5`, capped at `1.0` | |
| `attempt_pressure` | `(len(worker_outputs) - 1) / 4`, capped at `1.0` | first output is the initial attempt, not a retry |

Every extractor function defaults safely to a neutral value (`0.0`/`False`)
when the underlying data is absent, so an "empty" `AgentState` scores
predictably rather than raising.

### Scoring

Each candidate critic has a fixed weight table (`_CRITIC_WEIGHTS`) mapping
feature names to weights that sum to `1.0`, so with every feature in
`[0, 1]` the resulting score is always in `[0.0, 1.0]`:

| Critic | Dominant weights | Rationale |
|---|---|---|
| `LogicCritic` | `uncertainty` 0.35, `task_complexity` 0.30, `risk` 0.25 | general-purpose reasoning critic, sensitive to ambiguity and complexity |
| `CodeCritic` | `is_code_output` 0.50, `task_complexity` 0.25 | strongly favored when the output looks like code |
| `FactCritic` | `memory_relevance` 0.40, `uncertainty` 0.30 | favored when grounding in retrieved context matters and confidence is low |
| `MetaCritic` | `requires_meta_critic` 0.35, `iteration_pressure` 0.25, `requires_self_correction` 0.15, `attempt_pressure` 0.15 | favored as retries/escalation signals accumulate |

An unrecognized critic name falls back to `_DEFAULT_WEIGHTS`, a balanced
generic profile, rather than erroring.

These weights are the heuristic itself — a hand-authored linear scoring
function, the classic form of a deterministic "expert system" heuristic.
Nothing here is fit to data or updated based on outcomes.

## `CriticRanking`

Takes the `dict[str, float]` produced by the scorer and sorts it highest
score first. Ties are broken by critic name, ascending alphabetically, so
the ranking is fully deterministic regardless of the input dict's
iteration order. Exposes `top(n)`, `critic_names()`, `score_for(name)`,
and `as_list_of_dicts()` for diagnostics.

## `CriticSelector`

Three deterministic strategies over a `CriticRanking`, all via a single
`select(ranking, strategy, *, k=None, threshold=None)` dispatch method as
well as individually:

- `select_top_1` — the single highest-ranked critic.
- `select_top_k(ranking, k)` — the `k` highest-ranked critics.
- `select_by_threshold(ranking, threshold)` — every critic scoring `>= threshold`.

## Integration with `policy_engine_node`

`app/graph/nodes.py::policy_engine_node` delegates to this module:

```
candidate_critics = list(_CANDIDATE_CRITIC_NAMES)          # step 4
scores = HeuristicPolicyScorer().score(state, candidates)   # step 5
ranking = CriticRanking(scores)                              # step 6 (rank)
selected = CriticSelector().select(ranking, TOP_1)            # step 6 (select a*)
```

The full computation is written to
`state.memory_context["policy_engine"]` for diagnostics:
`candidate_critics`, `scores`, `ranking`, `selection_strategy`, and
`selected_critics`.

**This selection is diagnostic only.** `policy_engine_node` does **not**
write to `state.selected_critics` or `state.policy_decision` — those
remain `router_node`'s existing, unmodified responsibility (a simple
`task_type`-based rule). Since this scorer never reads `task_type`, its
own top pick can legitimately differ from what `router_node` selects; see
`tests/test_pipeline_integration.py::test_policy_engine_diagnostics_do_not_override_router_selection`.

## Explicit non-goals

- No reinforcement learning, Q-learning, PPO, or DQN.
- No neural networks or any learned/fitted parameters.
- No LLM calls, no API calls.
- No change to graph topology (`app/graph/state_graph.py`,
  `app/graph/edges.py` are untouched).
- No change to `router_node` or to what determines
  `state.selected_critics` / `state.policy_decision`.
- No randomness — every run of the same input is bit-for-bit identical.
