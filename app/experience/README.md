# Experience Memory Layer

This module records every completed ACRF execution as a reusable,
structured `ExperienceRecord`, so a future adaptive-learning algorithm
can consume a history of executions without any change to this module.

> **Scope:** this module only *collects* execution experiences. No
> reinforcement learning, no contextual bandits, no Q-learning, no
> PPO/DQN, no neural networks, no LLM calls, no training, no replay
> buffers, no RL datasets. `ExperienceRecorder` reads only `AgentState`
> and performs no business logic, no scoring, and no routing.

## Files

| File | Responsibility |
|---|---|
| `models.py` | `ExperienceRecord` — the frozen, self-describing snapshot of one execution. |
| `repository.py` | `ExperienceRepository` (abstract interface) and `InMemoryExperienceRepository` (the only concrete implementation here). |
| `recorder.py` | `ExperienceRecorder` — builds an `ExperienceRecord` from `AgentState`, optionally storing it into an injected repository. |

## `ExperienceRecord`

A frozen (`model_config = ConfigDict(extra="allow", frozen=True)`)
Pydantic v2 model with every field the task specifies:
`experience_id`, `session_id`, `task_id`, `timestamp`, `state_features`,
`selected_critics`, `critic_scores`, `aggregated_quality_score`,
`correction_decision`, `iterations`, `final_response`,
`execution_status`, `latency`, `estimated_cost`, `memory_usage`,
`metadata`.

Frozen because an experience is a historical fact about a completed
execution — once recorded, it should not be mutated in place.

### Field derivations

Every value is either copied directly from `AgentState`, or derived by a
purely mechanical operation — never a decision, score, or business rule:

- **`experience_id`** — `sha256(f"{session_id}|{task_id}|{iterations}")`.
  Deterministic (same triple → same id) and collision-resistant
  (different triples → different ids), with no randomness and no
  timestamp involved.
- **`timestamp`** — read directly from
  `state.execution_metadata.updated_at` (set by `evaluation_node`
  earlier in the same call), rather than calling the system clock
  independently. This keeps the recorder a pure function of its input:
  the same `AgentState` always yields the same `ExperienceRecord`.
- **`latency`** — `(execution_metadata.updated_at -
  execution_metadata.created_at).total_seconds()`: a timestamp
  subtraction, not a measurement mechanism of its own.
- **`estimated_cost`** — the sum of `token_usage` (an additive
  `WorkerOutput` field) across `state.worker_outputs`. **This is a
  placeholder proxy, not a real cost** — no pricing model exists yet in
  this framework, and `worker_node`'s `token_usage` is currently always
  `0`. Consumers should not treat this as an actual monetary figure.
- **`state_features`** — a snapshot dict (`task_type`, iteration counts,
  the full `error_features` dump, the `planner_output` dump). Pure
  transcription of already-computed data; no new features are derived.
- **`memory_usage`** — `{"retrieved_memories_count": ..., "memory_context_keys":
  ...}`, a snapshot of what's already on `state`.
- **`correction_decision`** — read from
  `state.memory_context["correction_policy"]["decision"]` if present
  (see `app/correction_policy`), else `None`.

## `ExperienceRepository`

An abstract interface (`add`, `get`, `list`, `clear`, `count`) with one
concrete implementation, `InMemoryExperienceRepository`: a plain dict
keyed by `experience_id`, alive only for the lifetime of the Python
process. No persistence, no database, no ChromaDB, no SQLite.

`add` **rejects** a record whose `experience_id` already exists
(`ValueError`) rather than silently overwriting it — a second layer of
duplicate protection beyond `experience_id`'s own collision resistance.

### Future compatibility

`ExperienceRecorder` depends only on the abstract `ExperienceRepository`
interface, never on `InMemoryExperienceRepository` directly. A future
`SqliteExperienceRepository`, `ChromaExperienceRepository`, or
`PostgresExperienceRepository` need only implement the same five methods
and can be substituted with **zero changes to `ExperienceRecorder`** —
exactly the requirement this module was designed around.

## `ExperienceRecorder`

```python
class ExperienceRecorder:
    def __init__(self, repository: ExperienceRepository | None = None) -> None: ...
    def record(self, state: AgentState) -> ExperienceRecord: ...
```

`repository` is injected via the constructor (dependency injection): pass
none to only build records, or pass any `ExperienceRepository` to also
store every built record into it. `record` reads only `AgentState` and
never mutates it.

## Integration with `evaluation_node`

`app/graph/nodes.py::evaluation_node` — and only this node — now, at the
very end of its existing logic (after `final_response`,
`execution_status`, and `execution_metadata` are set), does:

```python
experience = ExperienceRecorder(repository=DEFAULT_EXPERIENCE_REPOSITORY).record(state)
state.memory_context = {**state.memory_context, "experience": experience.model_dump(mode="json")}
```

`DEFAULT_EXPERIENCE_REPOSITORY` is a process-wide `InMemoryExperienceRepository`
singleton exported by this module. `evaluation_node` is a plain function
invoked by LangGraph with only `(state)` — there is no constructor or
call site through which a repository could otherwise be injected into
it — so a module-level singleton is the practical mechanism for giving
the node a repository that persists across calls within a process.
**Tests should construct and inject their own
`InMemoryExperienceRepository`** rather than relying on this shared
singleton, to stay isolated from each other.

No other node, and no graph topology, routing, policy, or critic code,
is touched.

## Explicit non-goals

- No reinforcement learning, contextual bandits, Q-learning, PPO, or DQN.
- No neural networks.
- No LLM calls.
- No training, no replay buffers, no RL dataset construction.
- No persistence, no database, no ChromaDB, no SQLite — in-memory only.
- No change to graph topology, routing, policy, or critic behavior.
- No business logic, scoring, or decision-making inside `ExperienceRecorder`.
