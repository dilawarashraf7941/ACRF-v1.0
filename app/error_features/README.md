# Error Feature Extractor — Data Models

This module defines the **data contract** for error features in ACRF: the
structured, domain-independent representation of "what went wrong" that
sits between raw execution signals (worker outputs, exceptions, critic
feedback, etc.) and the adaptive routing policy that decides what to do
about it.

> **Scope:** this module contains **only Pydantic v2 models**. There is no
> extraction logic, no classifiers, no heuristics, no LLM calls, and no
> routing decisions here. Those belong to future modules (e.g. an
> `extractor.py` that populates these models, and `app/router/` that
> consumes them).

## Why a dedicated module

The routing policy should never need to know *how* an error was detected —
only a consistent, typed shape it can reason over. Centralizing that shape
here means:

- Extractors (rule-based, statistical, model-based, or hybrid) can evolve
  independently as long as they emit `ErrorFeatureProfile` objects.
- Routing policies, critics, and meta-critics can be built and tested
  against a stable schema before any extraction logic exists.
- New error taxonomies, task domains, or critic types can be introduced
  without changing the schema, because open-ended concepts are modeled as
  free-form strings/tags rather than closed enums (see below).

## Models

### `ErrorFeatureProfile`

The canonical unit describing a single detected error. Fields fall into
four groups:

| Group | Fields |
|---|---|
| Error identity | `error_type`, `error_severity`, `confidence_score` |
| Task context | `task_category`, `task_complexity` |
| Risk & handling signals | `risk_level`, `required_expertise`, `suggested_critics`, `requires_meta_critic`, `requires_self_correction`, `memory_relevance`, `estimated_correction_difficulty` |
| Provenance | `extraction_metadata` |

### `ErrorFeatureCollection`

A reusable container grouping multiple `ErrorFeatureProfile` entries
detected for the same task/execution step, plus an optional
externally-assigned `overall_risk_level`. This lets a routing policy
reason over several simultaneous or accumulated errors without every
consumer inventing its own list wrapper.

### `ErrorFeatureExtractionMetadata`

Provenance/bookkeeping for a feature profile: which extractor produced it,
when, from which node, and which raw signal sources were considered. This
keeps auditability and reproducibility concerns out of the core feature
fields.

## Enums vs. free-form fields

Enums are used **only** where the concept is a small, closed, ordinal
scale that is meaningful across any domain:

- `ErrorSeverity` — low / medium / high / critical
- `RiskLevel` — low / medium / high / critical
- `TaskComplexity` — trivial / simple / moderate / complex / very_complex
- `CorrectionDifficulty` — trivial / easy / moderate / hard / very_hard

Everything that is inherently open-ended or domain-specific is deliberately
**left as a string or list of strings** instead of an enum, so the schema
never has to change as new domains, taxonomies, or critics are introduced:

- `error_type` — the space of possible errors is unbounded and
  domain-dependent (e.g. `"factual_inconsistency"`, `"timeout"`,
  `"schema_violation"`).
- `task_category` — task taxonomies vary per deployment.
- `required_expertise` — an open tag list (e.g. `"security"`,
  `"mathematics"`).
- `suggested_critics` — critic identifiers are defined by whatever critics
  exist in a given deployment, not by this schema.

## Numeric fields

- `confidence_score` (0.0–1.0) — how confident the (future) extractor is
  that the error was correctly identified.
- `memory_relevance` (0.0–1.0) — how relevant retrieving prior memory
  context is expected to be for resolving the error.

Both are constrained to `[0.0, 1.0]` at the schema level; no scoring
algorithm is implemented here.

## Extensibility

- Every model sets `model_config = ConfigDict(extra="allow")`, so future
  extractors or routing policies can attach additional fields without a
  breaking schema change.
- `extraction_metadata.extra` and `ErrorFeatureCollection.metadata` provide
  explicit, structured escape hatches for anything not yet promoted to a
  first-class field.

## Relationship to `app/state`

`app/state/state.py` defines a lightweight `ErrorFeature` model used
directly inside `AgentState.error_features`. `ErrorFeatureProfile` in this
module is the richer, dedicated representation intended for the
extraction/routing subsystem. Wiring the two together (e.g. having the
extractor populate `AgentState.error_features` from an
`ErrorFeatureCollection`) is a future integration step, not part of this
module's scope.

## Explicit non-goals

- No extraction algorithms or heuristics.
- No classifiers or scoring models.
- No LLM calls.
- No routing logic or policy decisions.
- No mutation of `AgentState`.

These are intentionally deferred to later modules that will consume the
types defined here.
