# Metrics & Experiment Framework

This module collects standardized evaluation metrics for research
experiments: one `ExecutionMetrics` per completed execution, aggregated
into an `ExperimentSummary` for comparing runs — across the current
Heuristic Policy or any future Contextual Bandit / Offline RL / PPO /
Q-learning policy — without requiring any change to this module.

> **Scope:** deterministic extraction and arithmetic only. No
> reinforcement learning, no contextual bandits, no policy optimization,
> no router/policy/critic/reward/experience changes, no LLM calls. No
> randomness anywhere in this module. Identical inputs always produce
> identical `ExecutionMetrics`/`ExperimentSummary`.

## Files

| File | Responsibility |
|---|---|
| `models.py` | `ExecutionMetrics` (one run) and `ExperimentSummary` (aggregate). |
| `collector.py` | `MetricsCollector` — pure extraction: `(AgentState, ExperienceRecord, RewardSignal) -> ExecutionMetrics`. |
| `aggregator.py` | `MetricsAggregator` — pure arithmetic: `list[ExecutionMetrics] -> ExperimentSummary`. |
| `repository.py` | `MetricsRepository` (abstract) and `InMemoryMetricsRepository` (the only concrete implementation here). |

## `ExecutionMetrics`

A frozen (`extra="allow", frozen=True`) Pydantic v2 model — the same
"immutable historical record" pattern used by `ExperienceRecord` and
`RewardSignal`: `execution_id`, `reward`, `aggregated_quality_score`,
`iterations`, `latency`, `estimated_cost`, `selected_critics`,
`correction_applied`, `execution_status`, `timestamp`, `metadata`.

## `MetricsCollector`

Reads only its three inputs and copies values across — **no
calculations** other than extracting values (a `len(...) > 0` presence
check and a `list(...)` copy are the only non-trivial operations):

| `ExecutionMetrics` field | Source |
|---|---|
| `execution_id` | `experience.experience_id` (shared with the `ExperienceRecord`) |
| `reward` | `reward.reward` |
| `aggregated_quality_score`, `iterations`, `latency`, `estimated_cost`, `selected_critics`, `execution_status`, `timestamp` | copied directly from `experience` |
| `correction_applied` | `len(state.correction_history) > 0` |

### Why `correction_applied` reads `state.correction_history`, not `experience.correction_decision`

`experience.correction_decision` only reflects the *last*
`self_correction_node` call's decision. If a run looped through
`self_correction_node` more than once (`SELF_CORRECTION_PATH_MAP`'s
`"retry"` edge), an earlier iteration could have genuinely applied a
correction while the *final* decision was `should_correct=False` (e.g. a
hard stop from `max_iterations_reached`). `state.correction_history` is
the complete, accumulated record of every correction actually applied
during the run, so checking whether it's non-empty is the accurate
signal for "was correction applied at any point," not just "what did the
last decision say."

### `metadata["policy"]`

There is no `policy` field in the requested `ExecutionMetrics` schema, so
the policy tag used for `average_reward_per_policy` /
`policy_usage` grouping lives in `metadata["policy"]`. `MetricsCollector`
reads `state.memory_context["policy_engine"]["policy_name"]` if a future
policy node ever publishes one, and otherwise defaults to
`DEFAULT_POLICY_NAME = "HeuristicPolicy"` (the only policy implementation
that exists today — see `app/policy_engine`). This is how the module
achieves **"must work unchanged for Heuristic Policy, Contextual
Bandits, Offline RL, PPO, Q-learning"**: a future policy node only needs
to publish its own name into that memory_context key, and
`MetricsCollector` picks it up automatically — no change to this module
required.

## `MetricsAggregator`

Every statistic is a plain mean, rate, or count:

| Field | Computation |
|---|---|
| `total_runs` | `len(metrics)` |
| `average_reward` | mean of every `reward` |
| `average_quality` | mean of `aggregated_quality_score` **excluding** `None` values |
| `average_iterations` | mean of every `iterations` |
| `average_latency` / `average_cost` | mean of `latency` / `estimated_cost`, excluding `None` |
| `success_rate` | fraction with `execution_status == "completed"` |
| `correction_rate` | fraction with `correction_applied == True` |
| `average_reward_per_policy` | mean `reward`, grouped by `metadata["policy"]` |
| `critic_selection_frequency` | count of each critic identifier's appearances across every `selected_critics` |
| `policy_usage` | count of runs per distinct `metadata["policy"]` |

### Graceful handling of empty input

`MetricsAggregator.aggregate([])` returns `total_runs=0`, every
average/rate as `None` (never a misleading `0.0`), and every dict
statistic as `{}` — verified explicitly in tests. The same graceful
`None`-for-missing-data behavior applies field-by-field even with
non-empty input: e.g. if no run recorded a `latency`, `average_latency`
is still `None`.

## `MetricsRepository`

An abstract interface (`add`, `list`, `clear`, `count`, `summary`) with
one concrete implementation, `InMemoryMetricsRepository`: a plain Python
list, alive only for the lifetime of the Python process. No persistence,
no database. `summary()` delegates to an injected `MetricsAggregator`
(dependency injection), defaulting to a plain one.

### Future compatibility

`MetricsCollector` and `MetricsAggregator` depend only on the plain
`ExecutionMetrics`/`ExperimentSummary` models and the abstract
`MetricsRepository` interface — never on `InMemoryMetricsRepository`
directly. A future `SqliteMetricsRepository`, `ChromaMetricsRepository`,
or `PostgresMetricsRepository` need only implement the same five methods
and can be substituted with **zero changes** to `MetricsCollector` or
`MetricsAggregator`.

## Integration with `evaluation_node`

`app/graph/nodes.py::evaluation_node` — and only this node — now, right
after computing the `RewardSignal` (see `app/reward`):

```python
metrics = MetricsCollector(repository=DEFAULT_METRICS_REPOSITORY).collect(
    state, enriched_experience, reward
)
state.memory_context["metrics"] = metrics.model_dump(mode="json")
```

`DEFAULT_METRICS_REPOSITORY` is a process-wide `InMemoryMetricsRepository`
singleton exported by this module, mirroring
`DEFAULT_EXPERIENCE_REPOSITORY`. No other node, and no graph topology,
routing, policy, critic, reward, or experience code, is touched.

## Explicit non-goals

- No reinforcement learning, contextual bandits, or policy optimization.
- No router, policy, critic, reward, or experience changes.
- No graph topology changes.
- No LLM calls.
- No persistence, no database — in-memory only.
- No calculations inside `MetricsCollector` beyond extracting values.
- No randomness — every run of the same input is bit-for-bit identical.
