"""Data models for the Adaptive Policy Engine in ACRF.

This module defines the structured, implementation-independent
representation of a routing policy: the versioned bundle of thresholds,
budgets, strategies, and sub-policies that the (future) policy engine
produces to guide the (future) router. It contains only Pydantic v2
models — no policy algorithms, no reinforcement learning, no routing
logic, and no LLM calls.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PolicyStrategy(str, Enum):
    """The high-level archetype a policy declares itself as following.

    This labels *what kind* of decision-making approach a policy
    represents; it does not implement that approach.
    """

    STATIC = "static"
    ADAPTIVE = "adaptive"
    EXPLORATORY = "exploratory"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    HYBRID = "hybrid"
    CUSTOM = "custom"


class ObjectiveType(str, Enum):
    """A small, domain-independent taxonomy of optimization objectives."""

    MAXIMIZE_QUALITY = "maximize_quality"
    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_LATENCY = "minimize_latency"
    MAXIMIZE_SAFETY = "maximize_safety"
    BALANCED = "balanced"
    CUSTOM = "custom"


class BackoffStrategy(str, Enum):
    """The shape of delay applied between retry attempts."""

    NONE = "none"
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


class MemoryScope(str, Enum):
    """The scope from which memory should be read/written under a policy."""

    NONE = "none"
    TASK = "task"
    SESSION = "session"
    GLOBAL = "global"


class SafetyEnforcementLevel(str, Enum):
    """How strictly a policy enforces safety findings."""

    ADVISORY = "advisory"
    STRICT = "strict"
    BLOCKING = "blocking"


class RoutingObjective(BaseModel):
    """The optimization objective(s) a policy is pursuing.

    Modeled as a structured object rather than a single field so a policy
    can express a primary objective, optional secondary objectives, and
    arbitrary named weights, without the schema assuming any particular
    multi-objective algorithm.
    """

    model_config = ConfigDict(extra="allow")

    primary: ObjectiveType = Field(
        ...,
        description="The primary optimization objective this policy pursues.",
    )
    secondary: list[ObjectiveType] = Field(
        default_factory=list,
        description="Additional objectives this policy considers secondary to the primary objective.",
    )
    weights: dict[str, float] = Field(
        default_factory=dict,
        description="Optional named weights (e.g. 'quality': 0.7, 'cost': 0.3) for objectives, including custom objectives beyond the fixed enum. No aggregation logic is implied here.",
    )
    description: str | None = Field(
        default=None,
        description="Free-form human-readable explanation of the objective, for documentation/audit purposes.",
    )


class CriticPriority(BaseModel):
    """The relative priority assigned to a single critic under a policy."""

    model_config = ConfigDict(extra="allow")

    critic_id: str = Field(
        ...,
        description="Identifier of the critic this priority entry applies to.",
    )
    priority: int = Field(
        ...,
        description="Relative priority/rank for this critic (lower or higher values may indicate higher priority; the ordering convention is defined by the consuming router, not this schema).",
    )
    rationale: str | None = Field(
        default=None,
        description="Optional human-readable explanation for why this critic was assigned this priority.",
    )


class BudgetConstraint(BaseModel):
    """A generic, unit-agnostic resource budget (used for both cost and latency budgets)."""

    model_config = ConfigDict(extra="allow")

    limit: float | None = Field(
        default=None,
        description="The numeric budget limit. Interpretation depends on 'unit'. None means no limit is enforced.",
    )
    unit: str | None = Field(
        default=None,
        description="Free-form unit for 'limit' (e.g. 'usd', 'tokens', 'ms', 'seconds'), left open to remain implementation-independent.",
    )
    hard_limit: bool = Field(
        default=True,
        description="Whether exceeding this budget should be treated as a hard stop versus merely flagged for review.",
    )


class RetryPolicy(BaseModel):
    """Declarative retry behavior a policy expects the router to honor."""

    model_config = ConfigDict(extra="allow")

    max_retries: int = Field(
        default=0,
        ge=0,
        description="Maximum number of retry attempts permitted under this policy.",
    )
    backoff_strategy: BackoffStrategy = Field(
        default=BackoffStrategy.NONE,
        description="The shape of delay to apply between retry attempts.",
    )
    retry_on: list[str] = Field(
        default_factory=list,
        description="Open-ended list of condition/error-type tags that should trigger a retry (e.g. 'timeout', 'low_confidence').",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional retry configuration not captured by the fixed fields above.",
    )


class MemoryUsagePolicy(BaseModel):
    """Declarative configuration for how memory should be used under a policy."""

    model_config = ConfigDict(extra="allow")

    enabled: bool = Field(
        default=True,
        description="Whether memory retrieval/usage is enabled under this policy.",
    )
    scope: MemoryScope = Field(
        default=MemoryScope.SESSION,
        description="The scope from which memory should be read/written.",
    )
    retrieval_top_k: int | None = Field(
        default=None,
        ge=0,
        description="Maximum number of memory records to retrieve, if bounded.",
    )
    relevance_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score, from 0.0 to 1.0, a retrieved memory must meet to be used.",
    )
    write_back: bool = Field(
        default=False,
        description="Whether newly generated content should be persisted back to memory under this policy.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional memory-usage configuration not captured by the fixed fields above.",
    )


class MetaCriticTriggerPolicy(BaseModel):
    """Declarative conditions under which a meta-critic review should be triggered."""

    model_config = ConfigDict(extra="allow")

    enabled: bool = Field(
        default=False,
        description="Whether meta-critic review is enabled at all under this policy.",
    )
    trigger_on_low_confidence: bool = Field(
        default=False,
        description="Whether a low confidence score alone should trigger meta-critic review.",
    )
    confidence_trigger_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence threshold, from 0.0 to 1.0, below which meta-critic review is suggested, if 'trigger_on_low_confidence' is set.",
    )
    trigger_on_conflicting_feedback: bool = Field(
        default=False,
        description="Whether disagreement among critics should trigger meta-critic review.",
    )
    trigger_conditions: list[str] = Field(
        default_factory=list,
        description="Additional open-ended, named conditions that should trigger meta-critic review.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional meta-critic trigger configuration not captured by the fixed fields above.",
    )


class SafetyPolicy(BaseModel):
    """Declarative safety enforcement configuration."""

    model_config = ConfigDict(extra="allow")

    enforcement_level: SafetyEnforcementLevel = Field(
        default=SafetyEnforcementLevel.STRICT,
        description="How strictly safety findings are enforced under this policy.",
    )
    require_safety_check: bool = Field(
        default=True,
        description="Whether a safety check is required before this policy permits finalization.",
    )
    blocked_categories: list[str] = Field(
        default_factory=list,
        description="Open-ended list of named categories that should be blocked, defined externally to this schema.",
    )
    escalate_on_flag: bool = Field(
        default=True,
        description="Whether a flagged safety status should escalate handling rather than proceed normally.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional safety configuration not captured by the fixed fields above.",
    )


class PolicyConstraint(BaseModel):
    """A single named constraint a policy imposes, described declaratively.

    No expression language or evaluation logic is implied; this is a
    structured description for a future constraint-evaluation component to
    interpret.
    """

    model_config = ConfigDict(extra="allow")

    name: str = Field(
        ...,
        description="A short, unique-within-policy name identifying this constraint.",
    )
    description: str | None = Field(
        default=None,
        description="Human-readable explanation of what this constraint requires.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary named parameters that a future constraint evaluator would use to interpret this constraint.",
    )


class PolicyMetadata(BaseModel):
    """Bookkeeping metadata describing the provenance and lifecycle of a policy."""

    model_config = ConfigDict(extra="allow")

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp marking when this policy was created.",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp marking when this policy was last updated.",
    )
    source: str | None = Field(
        default=None,
        description="Origin of this policy definition (e.g. 'default_config', 'admin_override', 'experiment_42').",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Open-ended labels for organizing, filtering, or searching policies.",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional policy metadata not captured by the fixed fields above.",
    )


class AdaptivePolicy(BaseModel):
    """A single, versioned policy specification produced by the Adaptive Policy Engine.

    This is the artifact the (future) policy engine emits to guide the
    (future) router: a complete, self-describing bundle of strategy,
    objective, critic selection, thresholds, budgets, and sub-policies.
    It represents a policy decision as *data*; it implements no decision
    logic itself.
    """

    model_config = ConfigDict(extra="allow")

    # --- Identity ---
    policy_id: str = Field(
        ...,
        description="Unique identifier for this policy definition.",
    )
    policy_version: str = Field(
        ...,
        description="Version identifier for this policy (e.g. '1.0.0'), enabling multiple versions of a policy to coexist and be audited.",
    )
    policy_strategy: PolicyStrategy = Field(
        default=PolicyStrategy.STATIC,
        description="The high-level decision-making archetype this policy declares itself as following.",
    )

    # --- Objective ---
    routing_objective: RoutingObjective = Field(
        ...,
        description="The optimization objective(s) this policy pursues when guiding routing decisions.",
    )

    # --- Critic Selection ---
    candidate_critics: list[str] = Field(
        default_factory=list,
        description="Identifiers of all critics eligible for consideration under this policy.",
    )
    selected_critics: list[str] = Field(
        default_factory=list,
        description="Identifiers of the critics this policy has selected from the candidates for the current context.",
    )
    critic_priority: list[CriticPriority] = Field(
        default_factory=list,
        description="Relative priority assigned to individual critics (candidate or selected) under this policy.",
    )

    # --- Thresholds ---
    confidence_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence, from 0.0 to 1.0, required before this policy accepts an upstream signal (e.g. an error feature) as actionable.",
    )
    quality_threshold: float = Field(
        default=0.0,
        description="Minimum acceptable aggregated quality score under this policy. Left unbounded since the quality scale is defined by the critics in use, not by this schema.",
    )

    # --- Budgets ---
    cost_budget: BudgetConstraint | None = Field(
        default=None,
        description="Optional resource budget constraining acceptable cost under this policy.",
    )
    latency_budget: BudgetConstraint | None = Field(
        default=None,
        description="Optional resource budget constraining acceptable latency under this policy.",
    )

    # --- Sub-policies ---
    retry_policy: RetryPolicy = Field(
        default_factory=RetryPolicy,
        description="Declarative retry behavior expected under this policy.",
    )
    memory_usage_policy: MemoryUsagePolicy = Field(
        default_factory=MemoryUsagePolicy,
        description="Declarative memory usage configuration expected under this policy.",
    )
    meta_critic_trigger_policy: MetaCriticTriggerPolicy = Field(
        default_factory=MetaCriticTriggerPolicy,
        description="Declarative conditions under which meta-critic review is expected to trigger under this policy.",
    )
    safety_policy: SafetyPolicy = Field(
        default_factory=SafetyPolicy,
        description="Declarative safety enforcement configuration expected under this policy.",
    )

    # --- Constraints & Provenance ---
    policy_constraints: list[PolicyConstraint] = Field(
        default_factory=list,
        description="Additional named constraints this policy imposes, beyond the structured thresholds and budgets above.",
    )
    policy_metadata: PolicyMetadata = Field(
        default_factory=PolicyMetadata,
        description="Bookkeeping metadata describing the provenance and lifecycle of this policy.",
    )
