"""Shared LangGraph execution state for ACRF.

Defines the single `AgentState` model that flows through every node of the
graph (planner, workers, critics, router, memory, etc.). This module
contains only data structures — no agents, routing, or critic logic.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExecutionStatus(str, Enum):
    """Lifecycle status of a task's execution within the graph."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SafetyStatus(str, Enum):
    """Safety/guardrail classification of a task at a given point in time."""

    UNKNOWN = "unknown"
    SAFE = "safe"
    FLAGGED = "flagged"
    BLOCKED = "blocked"


class PlannerOutput(BaseModel):
    """Generic container for the output of the planning stage."""

    model_config = ConfigDict(extra="allow")

    summary: str | None = Field(
        default=None,
        description="High-level natural-language summary of the plan produced by the planner.",
    )
    steps: list[str] = Field(
        default_factory=list,
        description="Ordered list of planned steps or subtasks, expressed generically as strings.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary planner-specific metadata not captured by the fixed fields above.",
    )


class WorkerOutput(BaseModel):
    """A single unit of output produced by a worker/agent execution."""

    model_config = ConfigDict(extra="allow")

    worker_id: str = Field(
        ...,
        description="Identifier of the worker/agent that produced this output.",
    )
    content: Any = Field(
        default=None,
        description="The raw output produced by the worker; intentionally untyped to remain agent-agnostic.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata associated with this worker's execution (e.g. timing, tool calls).",
    )


class ErrorFeature(BaseModel):
    """A structured description of an error or failure signal detected during execution."""

    model_config = ConfigDict(extra="allow")

    error_type: str = Field(
        ...,
        description="Category/type of the detected error (e.g. 'timeout', 'validation_error', 'hallucination').",
    )
    description: str = Field(
        ...,
        description="Human-readable description of the error condition.",
    )
    severity: str | None = Field(
        default=None,
        description="Optional severity label for the error (e.g. 'low', 'medium', 'high', 'critical').",
    )
    source_node: str | None = Field(
        default=None,
        description="Name of the graph node where the error was detected, if known.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional structured context about the error.",
    )


class CriticFeedback(BaseModel):
    """Qualitative feedback produced by a single critic."""

    model_config = ConfigDict(extra="allow")

    critic_name: str = Field(
        ...,
        description="Identifier of the critic that produced this feedback.",
    )
    feedback: str = Field(
        ...,
        description="Natural-language feedback or critique produced by the critic.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional structured data attached to this critic's feedback.",
    )


class PolicyDecision(BaseModel):
    """The outcome of the adaptive routing policy's decision-making."""

    model_config = ConfigDict(extra="allow")

    action: str | None = Field(
        default=None,
        description="The action or decision chosen by the routing policy (e.g. 'retry', 'escalate', 'finalize'), expressed generically.",
    )
    target_node: str | None = Field(
        default=None,
        description="The next graph node execution has been routed to, if applicable.",
    )
    rationale: str | None = Field(
        default=None,
        description="Optional explanation for why this decision was made.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional structured data supporting the policy decision.",
    )


class RetrievedMemory(BaseModel):
    """A single memory record retrieved from the memory subsystem (e.g. ChromaDB)."""

    model_config = ConfigDict(extra="allow")

    memory_id: str = Field(
        ...,
        description="Unique identifier of the retrieved memory record.",
    )
    content: str = Field(
        ...,
        description="The textual content of the retrieved memory.",
    )
    relevance_score: float | None = Field(
        default=None,
        description="Similarity/relevance score assigned to this memory by the retrieval mechanism.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata associated with the retrieved memory (e.g. source, timestamp).",
    )


class CorrectionRecord(BaseModel):
    """A single entry describing a correction applied during a prior iteration."""

    model_config = ConfigDict(extra="allow")

    iteration: int = Field(
        ...,
        description="The iteration number during which this correction was applied.",
    )
    description: str = Field(
        ...,
        description="Human-readable description of the correction that was applied.",
    )
    applied_by: str | None = Field(
        default=None,
        description="Identifier of the component (agent, critic, or policy) that applied the correction.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional structured context about the correction.",
    )


class ExecutionMetadata(BaseModel):
    """Framework-level bookkeeping metadata about the execution of the graph."""

    model_config = ConfigDict(extra="allow")

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp marking when this execution state was first created.",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp marking when this execution state was last updated.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional execution metadata (e.g. model versions, latency, token usage).",
    )


class AgentState(BaseModel):
    """The complete, shared execution state that flows through the ACRF LangGraph graph.

    Every node in the graph (planner, workers, critics, router, memory, etc.)
    reads from and writes to this single state object. The schema is
    intentionally generic and permissive so that new agents, critics, and
    routing strategies can be added without requiring changes to the state
    definition itself.
    """

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    # --- Identity ---
    session_id: str = Field(
        ...,
        description="Unique identifier for the overarching user session this task belongs to.",
    )
    task_id: str = Field(
        ...,
        description="Unique identifier for this specific task/run within the session.",
    )

    # --- Input ---
    user_query: str = Field(
        ...,
        description="The original, raw natural-language query or instruction submitted by the user.",
    )
    task_type: str | None = Field(
        default=None,
        description="Classification/category of the task, used to inform routing and agent selection. Left as a generic string to remain extensible to any taxonomy.",
    )

    # --- Planning & Execution ---
    planner_output: PlannerOutput | None = Field(
        default=None,
        description="Structured output produced by the planning stage of the framework, if a planner has run.",
    )
    worker_outputs: list[WorkerOutput] = Field(
        default_factory=list,
        description="Ordered collection of outputs produced by worker/agent executions for this task.",
    )

    # --- Error Signals ---
    error_features: list[ErrorFeature] = Field(
        default_factory=list,
        description="Structured signals describing errors or failures detected during execution, used to inform critics and routing.",
    )

    # --- Critique ---
    selected_critics: list[str] = Field(
        default_factory=list,
        description="Identifiers of the critics selected to evaluate the current outputs.",
    )
    critic_feedback: list[CriticFeedback] = Field(
        default_factory=list,
        description="Qualitative feedback collected from each selected critic.",
    )
    critic_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Mapping of critic identifier to the numeric score that critic assigned.",
    )
    aggregated_quality_score: float | None = Field(
        default=None,
        description="Single overall quality score derived by aggregating individual critic scores.",
    )

    # --- Routing ---
    policy_decision: PolicyDecision | None = Field(
        default=None,
        description="The most recent decision made by the adaptive routing policy based on critic feedback and scores.",
    )

    # --- Memory ---
    memory_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary contextual data assembled from memory to inform the current task execution.",
    )
    retrieved_memories: list[RetrievedMemory] = Field(
        default_factory=list,
        description="Memory records retrieved from the vector store (e.g. ChromaDB) relevant to the current task.",
    )

    # --- Iteration Control ---
    correction_history: list[CorrectionRecord] = Field(
        default_factory=list,
        description="Chronological record of corrections applied across iterations of this task.",
    )
    iteration_count: int = Field(
        default=0,
        description="Number of refinement/execution iterations completed so far for this task.",
    )
    max_iterations: int = Field(
        default=10,
        description="Upper bound on the number of iterations permitted before the framework must terminate or escalate.",
    )

    # --- Safety & Evaluation ---
    safety_status: SafetyStatus = Field(
        default=SafetyStatus.UNKNOWN,
        description="Current safety/guardrail status of the task, as determined by safety checks.",
    )
    evaluation_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Arbitrary quantitative evaluation metrics collected for this task (e.g. accuracy, latency, cost).",
    )
    execution_metadata: ExecutionMetadata = Field(
        default_factory=ExecutionMetadata,
        description="Framework-level bookkeeping metadata about the execution of this task.",
    )

    # --- Output & Control Flow ---
    final_response: str | None = Field(
        default=None,
        description="The final response to be returned to the user once execution has concluded.",
    )
    current_node: str | None = Field(
        default=None,
        description="Identifier of the LangGraph node currently processing, or that last processed, this state; used for tracing and debugging.",
    )
    execution_status: ExecutionStatus = Field(
        default=ExecutionStatus.PENDING,
        description="Overall lifecycle status of this task's execution within the graph.",
    )
