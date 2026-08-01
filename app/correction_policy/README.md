# Correction Decision Policy

This module replaces the previous "always correct" placeholder behavior
in `self_correction_node` with a **deterministic, rule-based decision**
about whether self-correction should run at all.

> **Scope:** deterministic heuristics only. No reinforcement learning, no
> Q-learning, no PPO, no DQN, no neural networks, no LLM calls. Every
> threshold in `rules.py` is a fixed constant chosen at implementation
> time — there is no training, no gradient updates, and no randomness.
> Identical `AgentState` values always produce identical decisions.

## Files

| File | Responsibility |
|---|---|
| `decision.py` | `CorrectionDecision` (the output model) and `CorrectionDecisionEngine` (reads `AgentState`, evaluates every rule, combines them by fixed priority). |
| `rules.py` | Six independent, pure rule functions plus `RuleResult`, the small record each one returns. |

## Inputs

`CorrectionDecisionEngine.decide` reads only the six fields specified:
`aggregated_quality_score`, `critic_scores`, `iteration_count`,
`max_iterations`, `memory_context`, and `error_features` (specifically,
the latest error feature's nested `metadata["profile"]` dump, the same
convention `app/policy_engine/scorer.py` uses).

## Rules

Each rule in `rules.py` is a pure function over primitive inputs (not
`AgentState`), so it can be constructed and tested in isolation without
building a full state:

| # | Rule | Condition | Signal |
|---|---|---|---|
| 1 | `rule_low_aggregated_quality` | `aggregated_quality_score < 0.7` | `correct` |
| 2 | `rule_max_iterations_reached` | `iteration_count >= max_iterations` | `no_correct` (hard stop) |
| 3 | `rule_meta_critic_escalation` | `critic_scores["MetaCritic"] > 0.7` | `correct` |
| 4 | `rule_requires_self_correction` | latest error feature's `profile.requires_self_correction` is `True` | `correct` |
| 5 | `rule_all_critics_high_quality` | every value in `critic_scores` `> 0.7` | `no_correct` ("finish") |
| 6 *(extra)* | `rule_low_memory_relevance` | `memory_context["memory_relevance"] < 0.2` | `correct` |

Rule 6 is not one of the five examples in the original spec, but is
included so `memory_context` — one of the engine's declared inputs — is
genuinely consulted rather than accepted-and-ignored. It reuses the same
forward-compatible `memory_context["memory_relevance"]` hook already
established by `app/policy_engine/scorer.py`.

Every rule defaults to *not triggered* (`signal="neutral"`) when its
underlying data is absent (e.g. no critic scores yet), so an "empty"
`AgentState` never spuriously triggers a rule.

## Decision combination

`CorrectionDecisionEngine.decide` evaluates **all six rules
unconditionally** (they are cheap and pure) and then applies a fixed
priority order:

1. **Hard stop.** If `rule_max_iterations_reached` triggers,
   `should_correct=False` — `confidence=1.0`,
   `decision_strategy="hard_stop_max_iterations"` — regardless of any
   other signal. Retrying past the iteration budget is never an option.
2. **Correction required.** Otherwise, if any of
   `rule_low_aggregated_quality`, `rule_meta_critic_escalation`,
   `rule_requires_self_correction`, or `rule_low_memory_relevance`
   triggers, `should_correct=True`,
   `decision_strategy="rule_based_correction"`. `confidence` scales with
   how many of these four agree: `min(1.0, 0.5 + 0.25 * n)`.
3. **Finish.** Otherwise, if `rule_all_critics_high_quality` triggers,
   `should_correct=False`, `confidence=0.9`,
   `decision_strategy="rule_based_finish"`.
4. **No signal.** Otherwise, `should_correct=False`, `confidence=0.0`,
   `decision_strategy="default_no_signal"` — a conservative default when
   there simply isn't enough information yet.

`CorrectionDecision.triggered_rules` always lists **every** rule that
matched, even ones outranked by a higher-priority rule — e.g. if quality
is low *and* the iteration budget is exhausted, both
`"low_aggregated_quality"` and `"max_iterations_reached"` appear, even
though the hard stop determines the final `should_correct=False`. Full
per-rule results (including untriggered ones) are additionally recorded
under `CorrectionDecision.metadata["rule_results"]` for diagnostics.

## `CorrectionDecision`

A Pydantic v2 model (`extra="allow"`, consistent with every other public
model in this codebase):

- `should_correct: bool`
- `reason: str`
- `confidence: float` (bounded `0.0`-`1.0`)
- `decision_strategy: str`
- `triggered_rules: list[str]`
- `metadata: dict[str, Any]`

## Integration with `self_correction_node`

`app/graph/nodes.py::self_correction_node` now:

1. Calls `CorrectionDecisionEngine().decide(state)`.
2. Unconditionally records diagnostics under
   `state.memory_context["correction_policy"]`: `decision` (the full
   `CorrectionDecision` dump), `triggered_rules`, `confidence`, and
   `strategy`.
3. If `should_correct` is `False`, returns immediately — no
   `CorrectionRecord`, no new `WorkerOutput`, no `iteration_count`
   change.
4. If `should_correct` is `True`, applies the exact same fixed
   placeholder correction as before (unchanged `CorrectionRecord` and
   `WorkerOutput` construction): appends a `CorrectionRecord`, increments
   `iteration_count`, and appends a placeholder "corrected"
   `WorkerOutput`.

No other node, and no graph topology or routing logic, is touched.

## Explicit non-goals

- No reinforcement learning, Q-learning, PPO, or DQN.
- No neural networks or any learned/fitted parameters.
- No LLM calls.
- No change to graph topology (`app/graph/state_graph.py`,
  `app/graph/edges.py`) or to any conditional-edge function.
- No change to routing (`router_node`) or to any critic
  (`app/critics/`).
- No randomness — every run of the same input is bit-for-bit identical.
