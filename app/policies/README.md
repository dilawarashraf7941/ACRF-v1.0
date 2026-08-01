# Adaptive Policy Engine — Data Models

This module defines the **data contract** for policies in ACRF: the
structured, versioned, implementation-independent representation of the
rules, thresholds, budgets, and sub-policies that a (future) policy engine
produces to guide a (future) router.

> **Scope:** this module contains **only Pydantic v2 models**. There is no
> policy-selection algorithm, no reinforcement learning, no routing logic,
> and no LLM calls here. Those belong to future modules (e.g. an engine
> that *produces* `AdaptivePolicy` instances, and `app/router/` that
> *consumes* them).

## Why a dedicated module

Separating "what a policy is" from "how a policy is chosen or executed"
lets the rest of the framework be built and tested independently:

- A router can be developed and unit-tested against realistic
  `AdaptivePolicy` fixtures before any policy-learning logic exists.
- Different policy engines (static config, heuristic, learned, hybrid) can
  all target the same schema, so the router never needs to know which
  produced a given policy.
- Policies are versioned and self-describing, so decisions made under them
  can be audited after the fact.

## The core model: `AdaptivePolicy`

`AdaptivePolicy` is the single artifact the policy engine emits. It is a
complete, self-contained bundle of everything a router needs to act,
organized into six groups:

| Group | Fields |
|---|---|
| Identity | `policy_id`, `policy_version`, `policy_strategy` |
| Objective | `routing_objective` |
| Critic selection | `candidate_critics`, `selected_critics`, `critic_priority` |
| Thresholds | `confidence_threshold`, `quality_threshold` |
| Budgets | `cost_budget`, `latency_budget` |
| Sub-policies | `retry_policy`, `memory_usage_policy`, `meta_critic_trigger_policy`, `safety_policy` |
| Constraints & provenance | `policy_constraints`, `policy_metadata` |

### Identity & strategy

- `policy_id` / `policy_version` — allow multiple versions of a policy to
  coexist, be referenced by decisions made under them, and be rolled back.
- `policy_strategy` (`PolicyStrategy` enum) — labels the decision-making
  archetype a policy claims to follow (`static`, `adaptive`, `exploratory`,
  `conservative`, `aggressive`, `hybrid`, `custom`). This is a label, not
  an implementation.

### Objective

`routing_objective` is a nested `RoutingObjective` model rather than a
single field, because real routing goals are rarely one-dimensional. It
carries a `primary` objective (`ObjectiveType` enum), optional
`secondary` objectives, and an open `weights` dict so custom, named
objectives can be expressed without changing the schema. No aggregation
or scalarization algorithm is implied.

### Critic selection

- `candidate_critics` — everything this policy is willing to consider.
- `selected_critics` — what this policy has chosen for the current
  context.
- `critic_priority` — a list of `CriticPriority` entries (`critic_id`,
  `priority`, optional `rationale`) rather than a bare `dict[str, int]`,
  so each priority assignment can carry an explanation. The ordering
  convention (higher-is-more-important vs. lower-is-more-important) is
  intentionally left to the consuming router, not fixed by this schema.

### Thresholds

- `confidence_threshold` — bounded to `[0.0, 1.0]`, consistent with the
  confidence conventions already used in `app/error_features`.
- `quality_threshold` — deliberately **unbounded**, because quality scores
  are produced by whatever critics are configured and may not live on a
  `0–1` scale. Constraining it here would leak an assumption about critic
  implementation into the policy schema.

### Budgets

`cost_budget` and `latency_budget` both use the same generic
`BudgetConstraint` shape (`limit`, `unit`, `hard_limit`) rather than
separate cost- and latency-specific models. `unit` is a free-form string
(`"usd"`, `"tokens"`, `"ms"`, `"seconds"`, ...) so the schema never assumes
a particular cost accounting or timing scheme.

### Sub-policies

Each of these is a small, independently reusable model so it can also be
composed outside of `AdaptivePolicy` if a future component only needs one
slice of policy configuration:

- `RetryPolicy` — `max_retries`, `backoff_strategy` (enum), open-ended
  `retry_on` condition tags.
- `MemoryUsagePolicy` — `enabled`, `scope` (`MemoryScope` enum),
  `retrieval_top_k`, `relevance_threshold`, `write_back`.
- `MetaCriticTriggerPolicy` — `enabled`, boolean triggers for low
  confidence and conflicting feedback, a numeric threshold, and open-ended
  `trigger_conditions`.
- `SafetyPolicy` — `enforcement_level` (`SafetyEnforcementLevel` enum),
  `require_safety_check`, open-ended `blocked_categories`,
  `escalate_on_flag`.

### Constraints & provenance

- `policy_constraints` — a list of `PolicyConstraint` (`name`,
  `description`, `parameters`), a declarative placeholder for constraints
  a future evaluator will interpret. No expression language or evaluation
  logic is defined here.
- `policy_metadata` — a `PolicyMetadata` model (`created_at`, `updated_at`,
  `source`, `tags`, `extra`) for provenance and lifecycle bookkeeping,
  mirroring the metadata pattern used elsewhere in ACRF
  (`app/state/state.py`'s `ExecutionMetadata`,
  `app/error_features`'s `ErrorFeatureExtractionMetadata`).

## Enums vs. free-form fields

Enums are used only for small, closed, structural taxonomies that are
meaningful regardless of domain: `PolicyStrategy`, `ObjectiveType`,
`BackoffStrategy`, `MemoryScope`, `SafetyEnforcementLevel`.

Everything that is inherently open-ended is left as a string, list of
strings, or dict, so new critics, objectives, retry conditions, or
constraint types never require a schema change:

- `candidate_critics` / `selected_critics` / `critic_priority[].critic_id`
  — critic identifiers are defined by whatever critics exist in a given
  deployment.
- `retry_policy.retry_on`, `meta_critic_trigger_policy.trigger_conditions`,
  `safety_policy.blocked_categories` — open-ended condition/category tags.
- `routing_objective.weights` — supports naming objectives beyond the
  fixed `ObjectiveType` enum.

## Extensibility

Every model sets `model_config = ConfigDict(extra="allow")`, and most
carry an explicit `metadata`/`extra` dict, so future policy engines can
attach additional fields without a breaking schema change.

## Relationship to other ACRF modules

- `app/state/state.py` defines a lightweight `PolicyDecision` model
  (`action`, `target_node`, `rationale`, `metadata`) stored on
  `AgentState.policy_decision` — the *outcome* of applying a policy during
  one graph step. `AdaptivePolicy` in this module is the richer, versioned
  *policy specification* that a future policy engine would evaluate to
  produce that outcome. Linking `AgentState.policy_decision` to a specific
  `AdaptivePolicy.policy_id`/`policy_version` is a future integration
  step, not part of this module's scope.
- `app/error_features` defines `ErrorFeatureProfile`/`ErrorFeatureCollection`,
  which a future policy engine would read as input when deciding on
  `selected_critics`, `requires_meta_critic`-style triggers, etc. This
  module does not import from `app/error_features`, keeping the two
  independently reusable.

## Explicit non-goals

- No policy-selection or policy-learning algorithms.
- No reinforcement learning.
- No routing logic.
- No LLM calls.
- No constraint-evaluation or budget-enforcement logic.
- No mutation of `AgentState`.

These are intentionally deferred to later modules that will consume the
types defined here.
