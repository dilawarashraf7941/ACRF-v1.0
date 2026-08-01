# Policy Abstraction Layer

This module replaces `policy_engine_node`'s direct call to
`HeuristicPolicyScorer` with a **pluggable policy interface**:
`BasePolicy.select_action(context, candidate_critics) -> PolicyDecision`.
The node no longer knows or cares which concrete policy is behind that
call.

> **Scope:** an interface and a deterministic implementation only. No
> contextual bandit algorithm (LinUCB, Thompson Sampling, or otherwise),
> no reinforcement learning, no PPO, no neural networks, no policy
> learning, and no reward/experience updates are implemented anywhere in
> this module. `ContextualBanditPolicy` is a stub that raises
> `NotImplementedError`. Graph topology, `router_node`, and ownership of
> `selected_critics`/`policy_decision` on `AgentState` are unchanged.

## Files

| File | Responsibility |
|---|---|
| `models.py` | `PolicyDecision` — the uniform, immutable output every policy produces. |
| `base.py` | `BasePolicy` — the abstract `select_action` contract. No implementation. |
| `heuristic_policy.py` | `HeuristicPolicy` — the default policy; deterministic, feature-based, behavior-identical to the pre-refactor scorer. |
| `contextual_bandit_policy.py` | `ContextualBanditPolicy` — a stub reserving the slot for a future bandit algorithm. Always raises `NotImplementedError`. |
| `registry.py` | `PolicyRegistry` and the module-level `DEFAULT_POLICY_REGISTRY` singleton. |

## `PolicyDecision`

A Pydantic v2 model (`extra="allow"`, `frozen=True`, consistent with
every other historical-record model in this codebase):

- `selected_critics: list[str]`
- `scores: dict[str, float]`
- `ranking: list[dict[str, Any]]`
- `policy_name: str`
- `policy_version: str`
- `confidence: float` (bounded `0.0`-`1.0`)
- `metadata: dict[str, Any]`

## `BasePolicy`

```python
class BasePolicy(ABC):
    policy_name: str
    policy_version: str

    @abstractmethod
    def select_action(self, context: ContextVector, candidate_critics: list[str]) -> PolicyDecision: ...
```

Every current and future policy — `HeuristicPolicy`,
`ContextualBanditPolicy`, and an eventual `OfflineRLPolicy`/
`OnlineRLPolicy` — implements this same signature, so adding a new
policy never requires changing `policy_engine_node`.

## `HeuristicPolicy`

The same deterministic scoring `policy_engine_node` always used, now
reached through `BasePolicy`. It reuses, rather than duplicates, the
pre-existing pieces:

- `HeuristicPolicyScorer.score_critic` (`app/policy_engine/scorer.py`) —
  the exact weighted-sum formula. Made public (renamed from
  `_score_critic`) specifically so this module can call it; the formula
  and weight tables were not moved or copied.
- `CriticRanking` / `CriticSelector` (`app/policy_engine/`) — unchanged,
  imported directly.

`select_action` receives a `ContextVector`, not an `AgentState` —
`HeuristicPolicyScorer.score_critic` needs a `StateFeatures`, computed
from an `AgentState`. To bridge this without giving `select_action` a
back door to the original state, `app/context/encoder.py::ContextEncoder`
was extended (see `app/context/README.md`) with nine features that
exactly mirror `HeuristicPolicyScorer.extract_features`'s inputs
(`uncertainty`, `risk`, `task_complexity`, `memory_relevance`,
`requires_self_correction`, `requires_meta_critic`, `is_code_output`,
`iteration_pressure`, `attempt_pressure`). `_state_features_from_context`
reads those nine values back out of `context.features` and reconstructs
an equivalent `StateFeatures`. A parity test
(`tests/test_context_encoder.py::test_heuristic_scorer_parity_features_match_extract_features_exactly`)
confirms this reconstruction is exact, and
`tests/test_policy_heuristic_policy.py` confirms `HeuristicPolicy`'s
output matches the pre-refactor `HeuristicPolicyScorer.score` output
bit-for-bit for the same underlying state.

`confidence` is set to the top-ranked critic's score (`0.0` if
`candidate_critics` is empty). `metadata` carries `selection_strategy`
and `context_id`, so `policy_engine_node` can still populate its
diagnostics dict with the same shape it always has.

## `ContextualBanditPolicy`

A stub. `select_action` always raises `NotImplementedError` with a
message explaining that no bandit algorithm, exploration strategy, or
learning exists yet. It exists only so the `"ContextualBanditPolicy"`
name is already registered and importable ahead of a real
implementation.

## `PolicyRegistry`

A plain name -> `BasePolicy` lookup table:

- `register(policy, *, default=False)` — stores `policy` under
  `policy.policy_name`; the first policy ever registered becomes the
  default regardless of the flag.
- `get(policy_name)` — raises `KeyError` if unregistered.
- `list()` — registered names.
- `default_policy()` — raises `ValueError` if nothing has been
  registered yet.

`DEFAULT_POLICY_REGISTRY` is a module-level singleton, pre-populated
with `HeuristicPolicy()` (default) and `ContextualBanditPolicy()`
(non-default), mirroring the `DEFAULT_EXPERIENCE_REPOSITORY` /
`DEFAULT_METRICS_REPOSITORY` pattern used elsewhere: `policy_engine_node`
is a plain function LangGraph invokes with only `(state)`, so there is
no constructor call site to inject a registry into. Tests construct
their own `PolicyRegistry` instead of relying on this shared singleton.

## Integration with `policy_engine_node`

`app/graph/nodes.py::policy_engine_node` now:

1. Builds a `ContextVector` via `ContextEncoder().encode(state)`.
2. Retrieves the default policy via
   `DEFAULT_POLICY_REGISTRY.default_policy()`.
3. Calls `policy.select_action(context, candidate_critics)`.
4. Records the same `state.memory_context["policy_engine"]` diagnostics
   dict as before — `candidate_critics`, `scores`, `ranking`,
   `selection_strategy`, `selected_critics` — now sourced from the
   returned `PolicyDecision`.

No other node, and no graph topology or routing logic, is touched.
`router_node` and ownership of `selected_critics`/`policy_decision` on
`AgentState` are unchanged.

## Explicit non-goals

- No contextual bandit algorithm (LinUCB, Thompson Sampling, or
  otherwise).
- No reinforcement learning, PPO, or neural networks.
- No policy learning, reward updates, or experience updates.
- No change to graph topology (`app/graph/state_graph.py`,
  `app/graph/edges.py`) or to any conditional-edge function.
- No change to routing (`router_node`).
- No randomness — every run of the same input is bit-for-bit identical.
