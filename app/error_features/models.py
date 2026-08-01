"""Data models for extracted error features in ACRF.

This module defines the structured, domain-independent representation of
error features that downstream adaptive routing policies, critics, and
meta-critics reason over. It contains only Pydantic v2 models — no
extraction logic, no classifiers, no LLM calls, and no routing decisions.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ErrorSeverity(str, Enum):
    """Ordinal severity of a detected error, independent of any domain."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskLevel(str, Enum):
    """Ordinal risk level associated with an error, used to gate downstream handling."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskComplexity(str, Enum):
    """Ordinal estimate of how complex the underlying task is."""

    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class CorrectionDifficulty(str, Enum):
    """Ordinal estimate of how difficult correcting the error is expected to be."""

    TRIVIAL = "trivial"
    EASY = "easy"
    MODERATE = "moderate"
    HARD = "hard"
    VERY_HARD = "very_hard"


class ErrorFeatureExtractionMetadata(BaseModel):
    """Bookkeeping metadata describing how a feature set was produced.

    Kept generic so any future extraction mechanism (rule-based, statistical,
    model-based, hybrid, etc.) can populate it without changing the schema.
    """

    model_config = ConfigDict(extra="allow")

    extractor_name: str | None = Field(
        default=None,
        description="Identifier of the extractor implementation that produced these features (e.g. 'static_rules_v1').",
    )
    extractor_version: str | None = Field(
        default=None,
        description="Version identifier of the extractor implementation, for reproducibility and auditing.",
    )
    extracted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp marking when this feature set was extracted.",
    )
    source_node: str | None = Field(
        default=None,
        description="Identifier of the graph node or component that triggered extraction, if known.",
    )
    signal_sources: list[str] = Field(
        default_factory=list,
        description="Generic identifiers of the input signals considered during extraction (e.g. 'worker_output', 'stack_trace', 'critic_feedback'), without describing how they were used.",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional extraction metadata not captured by the fixed fields above.",
    )


class ErrorFeatureProfile(BaseModel):
    """A single, structured, domain-independent representation of an extracted error.

    This is the canonical unit consumed by adaptive routing policies to
    decide how to respond to a detected error, without the policy needing
    any knowledge of how the features were derived or what domain the task
    belongs to.
    """

    model_config = ConfigDict(extra="allow")

    # --- Error Identity ---
    error_type: str = Field(
        ...,
        description="Open-ended category/type of the detected error (e.g. 'factual_inconsistency', 'timeout', 'schema_violation'). Left as a free-form string so new error taxonomies never require a schema change.",
    )
    error_severity: ErrorSeverity = Field(
        default=ErrorSeverity.LOW,
        description="Ordinal severity of the detected error.",
    )
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence, from 0.0 to 1.0, that this error was correctly identified and characterized.",
    )

    # --- Task Context ---
    task_category: str | None = Field(
        default=None,
        description="Open-ended category of the task being performed when the error occurred (e.g. 'code_generation', 'summarization'). Free-form to remain domain-independent.",
    )
    task_complexity: TaskComplexity | None = Field(
        default=None,
        description="Ordinal estimate of the complexity of the underlying task, independent of the error itself.",
    )

    # --- Risk & Handling Signals ---
    risk_level: RiskLevel = Field(
        default=RiskLevel.LOW,
        description="Ordinal risk level associated with this error, used by routing policies to gate escalation or caution.",
    )
    required_expertise: list[str] = Field(
        default_factory=list,
        description="Open-ended list of expertise tags believed necessary to address this error (e.g. 'security', 'mathematics'), independent of any specific agent implementation.",
    )
    suggested_critics: list[str] = Field(
        default_factory=list,
        description="Identifiers of critics that this feature profile suggests may be relevant for evaluating or resolving the error, without mandating their use.",
    )
    requires_meta_critic: bool = Field(
        default=False,
        description="Whether this feature profile suggests a higher-order meta-critic should review the outcome.",
    )
    requires_self_correction: bool = Field(
        default=False,
        description="Whether this feature profile suggests the originating agent should attempt self-correction.",
    )
    memory_relevance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Score, from 0.0 to 1.0, indicating how relevant retrieving prior memory context is expected to be for resolving this error.",
    )
    estimated_correction_difficulty: CorrectionDifficulty | None = Field(
        default=None,
        description="Ordinal estimate of how difficult correcting this error is expected to be.",
    )

    # --- Provenance ---
    extraction_metadata: ErrorFeatureExtractionMetadata = Field(
        default_factory=ErrorFeatureExtractionMetadata,
        description="Metadata describing how and when this feature profile was produced.",
    )


class ErrorFeatureCollection(BaseModel):
    """A set of error feature profiles associated with a single task or execution step.

    Provided as a reusable container so routing policies can reason over
    multiple, possibly heterogeneous, errors detected for the same unit of
    work without each policy having to define its own aggregation shape.
    """

    model_config = ConfigDict(extra="allow")

    features: list[ErrorFeatureProfile] = Field(
        default_factory=list,
        description="The individual error feature profiles detected for the associated task or execution step.",
    )
    overall_risk_level: RiskLevel | None = Field(
        default=None,
        description="An optional, externally-assigned aggregate risk level summarizing the collection as a whole.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional metadata describing the collection (e.g. how/when it was assembled).",
    )
