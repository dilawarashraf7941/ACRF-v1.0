# ACRF Execution Graph

This module defines the **structure** of the ACRF LangGraph execution
graph: which nodes exist, how they connect, and under what named
conditions execution branches or terminates.

> **Scope:** `state_graph.py` wires the nodes and conditional edges into a
> compilable `StateGraph(AgentState)`. Eight of the nine nodes in
> `nodes.py` are implemented as **deterministic placeholders** tracing
> Algorithm 1 (Adaptive Critic Routing) — no LLM calls, no ML, no
> learning, no adaptive intelligence; every value is fixed, heuristic, or
> echoed from input. Only `safety_node` remains an unimplemented
> placeholder that raises `NotImplementedError` (Algorithm 1 has no
> safety step). Every conditional-edge function in `edges.py` — the code
> that decides **which** node to go to next — is still unimplemented, so
> the compiled graph can execute nodes for real but cannot yet branch past
> `router` on its own; see `route_after_router` etc. below.

## Files

| File | Responsibility |
|---|---|
| `nodes.py` | Node functions (one per graph stage) and the `NodeName` enum identifying them. Eight are implemented deterministic placeholders; `safety_node` still raises `NotImplementedError`. |
| `edges.py` | Conditional-edge functions, their path maps, and the `TerminalCondition` enum. All four conditional-edge functions still raise `NotImplementedError` — only their path maps (the possible destinations) are declared. |
| `state_graph.py` | `build_graph()` / `compile_graph()` — assembles nodes and edges into a `StateGraph(AgentState)`. |

## State compatibility

The graph is built as `StateGraph(AgentState)`, using the shared model
from `app/state/state.py` directly as the graph's state schema. Every node
function has the signature `(state: AgentState) -> AgentState`, and every
conditional-edge function has the signature `(state: AgentState) -> str`,
so no adapter or translation layer is needed between the graph and the
frozen architecture models.

## Nodes

Nine nodes, declared in `nodes.py` and named via `NodeName`. Algorithm 1
numbers refer to Adaptive Critic Routing's ten steps:

| Node | Status | What it does |
|---|---|---|
| `planner` | Implemented (step 1) | Normalizes `user_query`, writes a fixed `PlannerOutput` (`original_query`, `normalized_query`, `task_type="general"`, `decomposition=[]`). |
| `worker` | Implemented (step 2) | Appends a fixed placeholder `WorkerOutput` (`output="Placeholder worker execution."`, `status="completed"`). |
| `error_feature_extractor` | Implemented (step 3) | Deterministic keyword/length heuristics classify the latest worker output (empty/short/normal, code/text); bridges an `ErrorFeatureProfile` (see `app/error_features`) into `state.error_features`. |
| `policy_engine` | Implemented (steps 4-6) | Builds one candidate `CriticAction` per built-in critic, scores each via `PlaceholderPolicyEngine` (always `0.0`, see `app/policies`), and selects a\* via a real `argmax` over those constant scores. Records everything under `state.memory_context["policy_engine"]`; does **not** write `selected_critics`/`policy_decision` — that stays `router`'s job. |
| `router` | Implemented | Simple rule: `task_type == "code"` → `["CodeCritic"]`, else `["LogicCritic"]`. Writes `selected_critics` and `policy_decision`. |
| `critic` | Implemented (steps 7-8) | Instantiates and runs each critic named in `selected_critics` (see `app/critics`; each critic's `evaluate` is itself a placeholder), then combines results via `MajorityVoteStrategy` (placeholder aggregation). Writes `critic_feedback`, `critic_scores`, `aggregated_quality_score`. |
| `self_correction` | Implemented (step 9, "correction required" branch) | Unconditionally logs a `CorrectionRecord`, increments `iteration_count`, and appends a new placeholder "corrected" `WorkerOutput`. Does **not** decide whether correction is needed — that's a routing decision, still unimplemented. |
| `safety` | **Not implemented** | Algorithm 1 has no safety step; raises `NotImplementedError`, unchanged. |
| `evaluation` | Implemented (step 10) | Sets `final_response` from the *latest* worker output (so it transparently reflects a prior correction), records fixed metrics, writes an execution trace to `execution_metadata.metadata["trace"]`, sets `execution_status=COMPLETED`. |

Every implemented node is still a **placeholder**: outputs are fixed,
heuristic, or echoed from input — never computed by a real scoring,
learning, or reasoning process. `policy_engine`'s "argmax" is real code,
but since every candidate's score is the same engine-supplied constant,
the selection is a deterministic function of candidate order, not of any
learned signal. See `app/graph/nodes.py` docstrings for the full
per-node rationale, and `tests/test_pipeline_integration.py` for the
end-to-end behavior this produces.

## Topology

```
START
  │
  ▼
planner ──▶ worker ──▶ error_feature_extractor ──▶ policy_engine ──▶ router
                                                                        │
                                        ┌───────────────┬──────────────┼───────────────┬────────────┐
                                        ▼               ▼              ▼                ▼            ▼
                                     worker          critic      self_correction      safety     evaluation
                                   (retry_worker) (evaluate_output) (apply_correction) (check_safety) (finalize)

critic ──▶ safety            (proceed_to_safety)
critic ──▶ self_correction   (needs_correction)

self_correction ──▶ worker      (retry)
self_correction ──▶ evaluation  (max_iterations_exceeded)
self_correction ──▶ END         (unrecoverable_error)

safety ──▶ evaluation        (safe)
safety ──▶ self_correction   (flagged)
safety ──▶ END                (safety_blocked)

evaluation ──▶ END
```

### Fixed transitions

`planner → worker → error_feature_extractor → policy_engine → router` is a
fixed pipeline: every task always passes through plan → work → extract →
evaluate policy before a routing decision is made. This ordering is
structural, not a routing algorithm — no decision is made along this
chain.

### Conditional transitions

Four branch points are defined via `add_conditional_edges`, each backed by
a placeholder function in `edges.py` that raises `NotImplementedError`,
together with a static path map describing the *shape* of the branch
(named outcome → destination node). The decision logic that will one day
choose among these named outcomes is explicitly out of scope here:

| Branch point | Function | Named outcomes → destination |
|---|---|---|
| `router` | `route_after_router` | `retry_worker → worker`, `evaluate_output → critic`, `apply_correction → self_correction`, `check_safety → safety`, `finalize → evaluation` |
| `critic` | `route_after_critic` | `proceed_to_safety → safety`, `needs_correction → self_correction` |
| `self_correction` | `route_after_self_correction` | `retry → worker`, `max_iterations_exceeded → evaluation`, `unrecoverable_error → END` |
| `safety` | `route_after_safety` | `safe → evaluation`, `flagged → self_correction`, `safety_blocked → END` |

## Terminal conditions

Execution reaches `END` under exactly three named conditions
(`TerminalCondition` in `edges.py`), plus normal completion:

1. **`completed`** — `evaluation` finishes and the graph reaches `END`
   via its unconditional edge. This is the normal, successful termination
   path (reached after `safety` resolves `safe`, or after
   `self_correction` resolves `max_iterations_exceeded` and evaluation
   still runs to produce a best-effort final response).
2. **`safety_blocked`** — the `safety` node determines output must not
   proceed; the graph terminates immediately without reaching
   `evaluation`.
3. **`unrecoverable_error`** — the `self_correction` node determines no
   further retry is viable; the graph terminates immediately.

The `iteration_count` / `max_iterations` guard that would distinguish
`retry` from `max_iterations_exceeded` inside `route_after_self_correction`
is **not implemented** — only the named outcome and its destination are
declared.

## Explicit non-goals

- No node performs real planning, generation, extraction, scoring,
  correction, safety assessment, or evaluation *intelligence* — every
  implemented node's output is fixed, heuristic, or echoed from its
  input, never learned or adaptively computed.
- No conditional-edge function contains routing/decision logic; all four
  still raise `NotImplementedError`, so *which* node runs next after a
  branch point is not yet decided anywhere in this module.
- No LLM calls anywhere in this module.
- No reinforcement learning or real adaptive policy algorithms.
- No prompts.
- No API calls.
- `safety_node` is unimplemented, since Algorithm 1 has no safety step.

These are deferred to future modules that implement real intelligence
behind each node and real decision logic behind each conditional-edge
function, against the interfaces and placeholder behavior established
here.
