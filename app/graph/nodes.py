"""Node interfaces for the ACRF LangGraph execution graph.

Each function below defines a graph node's interface — its name,
signature, and documented responsibility. `planner_node`, `worker_node`,
`error_feature_extractor_node`, `policy_engine_node`, `router_node`,
`critic_node`, `self_correction_node`, and `evaluation_node` are all
implemented as deterministic placeholders (no LLM calls, no ML, no
embeddings, no planning/reasoning/learning/adaptive intelligence). Only
`safety_node` remains an unimplemented placeholder that raises
`NotImplementedError`, since Algorithm 1 (Adaptive Critic Routing) has no
safety step; safety behavior is deferred to a future module.

Together, the implemented nodes trace Algorithm 1's ten steps:

    1. planner_node                 -- decompose the query
    2. worker_node                  -- generate an initial response r
    3. error_feature_extractor_node -- derive feature vector F
    4-6. policy_engine_node         -- build candidate critics A, encode
                                        state as a `ContextVector` (see
                                        `app/context`), and delegate
                                        scoring, ranking, and selection of
                                        a* to the default `BasePolicy`
                                        (see `app/policy`)
    7-8. critic_node                -- execute the critics selected by
                                        `router_node` and aggregate their
                                        results via a placeholder
                                        AggregationStrategy
    9. self_correction_node         -- once the graph routes execution
                                        here, consult
                                        `CorrectionDecisionEngine` (see
                                        `app/correction_policy`) and apply
                                        the fixed placeholder correction
                                        only if that deterministic policy
                                        calls for it
    10. evaluation_node             -- store an execution trace, finalize
                                        the response, record the completed
                                        execution as a reusable
                                        `ExperienceRecord` (see
                                        `app/experience`), convert it into
                                        a deterministic `RewardSignal` (see
                                        `app/reward`), and extract a
                                        standardized `ExecutionMetrics`
                                        (see `app/metrics`)

Steps 4-6 use real, deterministic, feature-based heuristics (see
`app/policy_engine`) rather than a constant placeholder score — different
`AgentState` values genuinely produce different scores and rankings. This
is still not adaptive/learned: every weight is a fixed constant chosen at
implementation time, with no training, gradient updates, or randomness
involved. Which node actually determines `state.selected_critics` (and
therefore what `critic_node` executes) remains `router_node`'s existing,
unmodified, simple task-type rule; `policy_engine_node`'s own
candidate/score/ranking/selection computation is recorded under
`state.memory_context["policy_engine"]` for traceability without
overwriting `router_node`'s output.

Step 9 similarly no longer *unconditionally* applies a correction:
`self_correction_node` now delegates the "should we correct?" question to
`CorrectionDecisionEngine`'s fixed rules (see `app/correction_policy`),
recording the decision under `state.memory_context["correction_policy"]`
on every call, and only performing the (unchanged) placeholder
`CorrectionRecord`/`WorkerOutput`/`iteration_count` update when the
policy's `should_correct` is `True`. This is still not learned — every
rule and threshold is a fixed constant.

`evaluation_node` additionally records every completed execution as an
`ExperienceRecord` via `ExperienceRecorder` (see `app/experience`), which
reads only `AgentState` and performs no business logic, learning,
scoring, or routing of its own — it exists purely to give future
adaptive-learning algorithms a structured execution history to consume.
Immediately after, `RewardCalculator` (see `app/reward`) converts that
record into a deterministic `RewardSignal` — fixed weights, no
reinforcement learning, no policy updates, no repository of its own —
purely so future contextual-bandit/offline-RL/PPO/Q-learning algorithms
have a ready-made reward to consume as-is. Finally, `MetricsCollector`
(see `app/metrics`) extracts a standardized `ExecutionMetrics` from all
three — no calculations beyond a presence check — so research experiments
can compare runs across the current Heuristic Policy or any future
learning policy without this module ever needing to change.

The conditional edge functions in `app/graph/edges.py` (which decide
*whether* to route to `critic`/`self_correction`/`safety`/`evaluation` at
all) are unchanged and still unimplemented — this module only implements
what each node *does* once reached, not the routing decisions that reach
it.

Each node function accepts and returns the shared `AgentState` (see
`app/state/state.py`), making every node structurally compatible with a
`langgraph.graph.StateGraph(AgentState)` graph.
"""

from datetime import datetime, timezone
from enum import Enum

from app.context import ContextEncoder
from app.correction_policy import CorrectionDecisionEngine
from app.critics import (
    BaseCritic,
    CodeCritic,
    CriticResult,
    FactCritic,
    LogicCritic,
    MajorityVoteStrategy,
    MetaCritic,
)
from app.error_features import (
    ErrorFeatureCollection,
    ErrorFeatureExtractionMetadata,
    ErrorFeatureProfile,
    RiskLevel,
)
from app.experience import DEFAULT_EXPERIENCE_REPOSITORY, ExperienceRecorder
from app.metrics import DEFAULT_METRICS_REPOSITORY, MetricsCollector
from app.policy import DEFAULT_POLICY_REGISTRY
from app.reward import RewardCalculator
from app.state import (
    AgentState,
    CorrectionRecord,
    CriticFeedback,
    ErrorFeature,
    ExecutionMetadata,
    ExecutionStatus,
    PlannerOutput,
    PolicyDecision,
    WorkerOutput,
)


class NodeName(str, Enum):
    """Canonical identifiers for every node in the ACRF execution graph.

    Used by `app/graph/state_graph.py` and `app/graph/edges.py` so node
    names are declared once and referenced consistently when wiring the
    graph.
    """

    PLANNER = "planner"
    WORKER = "worker"
    ERROR_FEATURE_EXTRACTOR = "error_feature_extractor"
    POLICY_ENGINE = "policy_engine"
    ROUTER = "router"
    CRITIC = "critic"
    SELF_CORRECTION = "self_correction"
    SAFETY = "safety"
    EVALUATION = "evaluation"


def _normalize_query(user_query: str) -> str:
    """Collapse leading/trailing/internal whitespace in a query.

    This is plain string normalization only — no NLP, no semantic
    interpretation, no planning intelligence.

    Args:
        user_query: The raw query to normalize.

    Returns:
        The query with runs of whitespace collapsed to single spaces and
        surrounding whitespace stripped.
    """
    return " ".join(user_query.split())


def planner_node(state: AgentState) -> AgentState:
    """Deterministically populate `state.planner_output` from `state.user_query`.

    This is a deterministic placeholder implementation: it performs no
    planning intelligence and makes no LLM calls. It only normalizes the
    raw query and records a fixed `PlannerOutput` so downstream nodes and
    tests have a well-formed value to work with. The output is written
    using `PlannerOutput`'s `extra="allow"` configuration, since the
    requested fields (`original_query`, `normalized_query`, `task_type`,
    `decomposition`, `planning_notes`) are additive to the frozen model's
    declared fields (`summary`, `steps`, `metadata`).

    Args:
        state: The current agent state. Must have `user_query` set.

    Returns:
        The same `AgentState` instance, with `planner_output` populated.
    """
    state.planner_output = PlannerOutput(
        original_query=state.user_query,
        normalized_query=_normalize_query(state.user_query),
        task_type="general",
        decomposition=[],
        planning_notes="Placeholder planner",
    )
    return state


def _resolve_worker_input(state: AgentState) -> str:
    """Determine the text a worker acts on.

    Prefers `state.planner_output.original_query` when available (an
    additive field permitted by `PlannerOutput`'s `extra="allow"`
    configuration, per the same pattern used in `planner_node`), and falls
    back to `state.user_query` otherwise. This is plain field selection —
    no reasoning is performed.

    Args:
        state: The current agent state.

    Returns:
        `state.planner_output.original_query` if `state.planner_output` is
        set and carries a non-`None` `original_query`; otherwise
        `state.user_query`.
    """
    planner_output = state.planner_output
    if planner_output is not None:
        original_query = getattr(planner_output, "original_query", None)
        if original_query is not None:
            return original_query
    return state.user_query


def worker_node(state: AgentState) -> AgentState:
    """Deterministically append a placeholder `WorkerOutput` to `state.worker_outputs`.

    This is a deterministic placeholder implementation: it performs no
    reasoning and makes no LLM calls. It records a fixed `WorkerOutput`
    describing the input it was given, so downstream nodes and tests have
    a well-formed value to work with. The output is written using
    `WorkerOutput`'s `extra="allow"` configuration, since most of the
    requested fields (`worker_name`, `worker_type`, `input`, `output`,
    `reasoning_summary`, `confidence`, `execution_time`, `token_usage`,
    `status`) are additive to the frozen model's declared fields
    (`worker_id`, `content`, `metadata`).

    Args:
        state: The current agent state. Must have `user_query` set, and
            may optionally have `planner_output` set.

    Returns:
        The same `AgentState` instance, with a new `WorkerOutput` appended
        to `worker_outputs`.
    """
    worker_output = WorkerOutput(
        worker_id="worker-001",
        worker_name="DefaultWorker",
        worker_type="general",
        input=_resolve_worker_input(state),
        output="Placeholder worker execution.",
        reasoning_summary="No reasoning performed.",
        confidence=1.0,
        execution_time=0.0,
        token_usage=0,
        status="completed",
        metadata={},
    )
    state.worker_outputs = [*state.worker_outputs, worker_output]
    return state


_SHORT_OUTPUT_MAX_LENGTH = 20
"""Maximum stripped-output character length still classified as 'short'."""

_CODE_KEYWORDS: tuple[str, ...] = (
    "def ",
    "class ",
    "function ",
    "import ",
    "return ",
    "public ",
    "private ",
    "static ",
    "console.log",
    "print(",
    "#include",
    "using namespace",
    "select ",
    "from ",
    "where ",
    "var ",
    "let ",
    "const ",
)
"""Common programming keywords used by the deterministic code-detection heuristic."""


def _get_latest_worker_output(state: AgentState) -> WorkerOutput | None:
    """Return the most recent `WorkerOutput`, or `None` if none exist yet.

    Args:
        state: The current agent state.

    Returns:
        `state.worker_outputs[-1]` if non-empty, otherwise `None`.
    """
    if state.worker_outputs:
        return state.worker_outputs[-1]
    return None


def _extract_output_text(worker_output: WorkerOutput | None) -> str:
    """Extract the text to run heuristics over from a worker output.

    Prefers the `output` field set by `worker_node` (an additive field
    permitted by `WorkerOutput`'s `extra="allow"` configuration), and
    falls back to the declared `content` field if it holds a string. This
    is plain field selection — no reasoning is performed.

    Args:
        worker_output: The worker output to read, or `None`.

    Returns:
        The text to analyze, or an empty string if none is available.
    """
    if worker_output is None:
        return ""
    output = getattr(worker_output, "output", None)
    if isinstance(output, str):
        return output
    if isinstance(worker_output.content, str):
        return worker_output.content
    return ""


def _contains_code_indicators(text: str) -> bool:
    """Detect whether `text` looks like it contains source code.

    Uses two simple, deterministic heuristics: the presence of a fenced
    code block (triple backticks) or any of a fixed list of common
    programming keywords, matched case-insensitively. No ML, no
    embeddings, no LLM calls.

    Args:
        text: The text to inspect.

    Returns:
        `True` if either heuristic matches, `False` otherwise.
    """
    if "```" in text:
        return True
    lowered = text.lower()
    return any(keyword in lowered for keyword in _CODE_KEYWORDS)


def error_feature_extractor_node(state: AgentState) -> AgentState:
    """Deterministically extract exactly six error features from the latest worker output.

    This is a deterministic, heuristic-only placeholder implementation: no
    LLM calls, no ML, no embeddings — only string inspection against fixed
    thresholds and keyword lists. It extracts:

    1. `task_type` — resolved via `_resolve_task_type` (`state.task_type`,
       falling back to `state.planner_output.task_type`).
    2. `output_type` — `"code"` if the output contains a fenced code block
       or a common programming keyword, else `"text"`.
    3. `confidence` — `0.0` if the output is empty, `0.4` if it is
       non-empty but shorter than `_SHORT_OUTPUT_MAX_LENGTH` characters,
       else `1.0`.
    4. `risk_level` — `HIGH` / `MEDIUM` / `LOW`, mirroring the
       empty/short/otherwise cases above.
    5. `error_category` — `"empty_output"` / `"short_output"` / `"none"`,
       mirroring the same cases.
    6. `suggested_critics` — `["CodeCritic"]` if code was detected, else
       `["LogicCritic"]`.

    These six features are set on an `ErrorFeatureProfile` (see
    `app/error_features`): populated onto its declared fields where the
    name and semantics match exactly (`risk_level`, `suggested_critics`),
    mirrored onto the closest declared analogs for internal consistency
    (`error_type`, `confidence_score`, `task_category`), and also added
    verbatim as additive fields (`task_type`, `output_type`, `confidence`,
    `error_category`) via the model's `extra="allow"` configuration, so
    all six requested names are available on the resulting object exactly
    as named. The profile is then appended to an `ErrorFeatureCollection`.

    `AgentState.error_features` is frozen as `list[ErrorFeature]` — the
    lighter-weight model declared in `app/state/state.py`, distinct from
    `ErrorFeatureProfile`. Assigning an `ErrorFeatureProfile` or
    `ErrorFeatureCollection` directly to `state.error_features` fails
    Pydantic validation, since neither is an instance of the declared
    `ErrorFeature` type. To respect the frozen state contract, this node
    bridges the extraction result into a new `ErrorFeature`
    (`error_type=error_category`, `severity=risk_level.value`,
    `source_node` set to this node's name) whose `metadata` carries the
    full six-feature summary plus complete dumps of both the
    `ErrorFeatureProfile` and the `ErrorFeatureCollection`, so no
    information is lost. That `ErrorFeature` is appended to
    `state.error_features`.

    Args:
        state: The current agent state. Reads the latest entry of
            `worker_outputs` (if any) and `task_type`/`planner_output` for
            task-type resolution.

    Returns:
        The same `AgentState` instance, with a new `ErrorFeature` appended
        to `error_features`.
    """
    latest_worker_output = _get_latest_worker_output(state)
    output_text = _extract_output_text(latest_worker_output)
    stripped_text = output_text.strip()

    is_empty = stripped_text == ""
    is_short = not is_empty and len(stripped_text) < _SHORT_OUTPUT_MAX_LENGTH
    is_code = _contains_code_indicators(output_text)

    if is_empty:
        confidence = 0.0
        risk_level = RiskLevel.HIGH
        error_category = "empty_output"
    elif is_short:
        confidence = 0.4
        risk_level = RiskLevel.MEDIUM
        error_category = "short_output"
    else:
        confidence = 1.0
        risk_level = RiskLevel.LOW
        error_category = "none"

    output_type = "code" if is_code else "text"
    suggested_critics = ["CodeCritic"] if is_code else ["LogicCritic"]
    task_type = _resolve_task_type(state)

    profile = ErrorFeatureProfile(
        error_type=error_category,
        confidence_score=confidence,
        task_category=task_type,
        risk_level=risk_level,
        suggested_critics=suggested_critics,
        extraction_metadata=ErrorFeatureExtractionMetadata(
            extractor_name="deterministic_heuristic_extractor_v1",
            source_node=NodeName.ERROR_FEATURE_EXTRACTOR.value,
            signal_sources=["worker_output"],
        ),
        task_type=task_type,
        output_type=output_type,
        confidence=confidence,
        error_category=error_category,
    )
    collection = ErrorFeatureCollection(features=[profile], overall_risk_level=risk_level)

    bridged_feature = ErrorFeature(
        error_type=error_category,
        description=(
            f"Deterministic heuristic extraction: error_category={error_category!r}, "
            f"output_type={output_type!r}, task_type={task_type!r}."
        ),
        severity=risk_level.value,
        source_node=NodeName.ERROR_FEATURE_EXTRACTOR.value,
        metadata={
            "task_type": task_type,
            "output_type": output_type,
            "confidence": confidence,
            "risk_level": risk_level.value,
            "error_category": error_category,
            "suggested_critics": suggested_critics,
            "profile": profile.model_dump(mode="json"),
            "collection": collection.model_dump(mode="json"),
        },
    )
    state.error_features = [*state.error_features, bridged_feature]
    return state


_CANDIDATE_CRITIC_NAMES: tuple[str, ...] = ("LogicCritic", "CodeCritic", "FactCritic", "MetaCritic")
"""The fixed, built-in critic identifiers considered as candidates by `policy_engine_node`."""


def policy_engine_node(state: AgentState) -> AgentState:
    """Score, rank, and select candidate critics via the default policy.

    Algorithm 1, steps 4-6.

    This delegates to `app/policy` — a pluggable `BasePolicy` interface —
    for scoring, ranking, and selection, rather than invoking any scoring
    logic directly. The active policy is `DEFAULT_POLICY_REGISTRY`'s
    default, which is `HeuristicPolicy`: the same deterministic,
    feature-based heuristics used before this policy abstraction existed
    (no LLM calls, no reinforcement learning, no neural networks, no
    randomness). Different `AgentState` values (via `error_features`,
    `memory_context`, `iteration_count`, `planner_output`, and
    `worker_outputs`) genuinely produce different scores; see
    `app/policy/heuristic_policy.py` for the scoring heuristics.

    - Step 4: the fixed candidate set A is `_CANDIDATE_CRITIC_NAMES` (one
      identifier per built-in critic).
    - Step 5-6: `ContextEncoder().encode(state)` builds a `ContextVector`,
      and `policy.select_action(context, candidate_critics)` scores,
      ranks, and selects a* in one call.

    The full computation (candidate critics, scores, ranking, the
    selection strategy used, and the resulting selection) is recorded
    under `state.memory_context["policy_engine"]` for traceability. This
    node deliberately does **not** write to `state.selected_critics` or
    `state.policy_decision` — those remain `router_node`'s existing,
    unmodified responsibility, so `critic_node` continues to execute
    exactly the critics `router_node` selects.

    Args:
        state: The current agent state. Reads `error_features`,
            `memory_context`, `iteration_count`, `planner_output`, and
            `worker_outputs` (via `ContextEncoder`).

    Returns:
        The same `AgentState` instance, with `state.memory_context`
        updated under the `"policy_engine"` key.
    """
    candidate_critics = list(_CANDIDATE_CRITIC_NAMES)

    context = ContextEncoder().encode(state)
    policy = DEFAULT_POLICY_REGISTRY.default_policy()
    decision = policy.select_action(context, candidate_critics)

    state.memory_context = {
        **state.memory_context,
        "policy_engine": {
            "candidate_critics": candidate_critics,
            "scores": decision.scores,
            "ranking": decision.ranking,
            "selection_strategy": decision.metadata.get("selection_strategy"),
            "selected_critics": decision.selected_critics,
        },
    }
    return state


def _resolve_task_type(state: AgentState) -> str | None:
    """Determine the task type used to drive critic selection.

    Prefers `state.task_type` (the frozen `AgentState` field intended for
    task classification) and falls back to `state.planner_output.task_type`
    (an additive field set by `planner_node`) when the former is unset.
    This is plain field selection — no classification or learning is
    performed.

    Args:
        state: The current agent state.

    Returns:
        The resolved task type, or `None` if neither source has one.
    """
    if state.task_type is not None:
        return state.task_type
    if state.planner_output is not None:
        return getattr(state.planner_output, "task_type", None)
    return None


def _select_critics_for_task_type(task_type: str | None) -> list[str]:
    """Select critics for a task type via a single fixed, deterministic rule.

    Args:
        task_type: The resolved task type, or `None` if unknown.

    Returns:
        `["CodeCritic"]` if `task_type == "code"`, otherwise
        `["LogicCritic"]`.
    """
    if task_type == "code":
        return ["CodeCritic"]
    return ["LogicCritic"]


def router_node(state: AgentState) -> AgentState:
    """Deterministically select critics and record a rule-based policy decision.

    This is a deterministic, rule-based placeholder implementation: it
    performs no learning and makes no LLM calls. It applies one fixed
    rule — task type `"code"` routes to `CodeCritic`, everything else
    routes to `LogicCritic` — to populate `state.selected_critics` and
    `state.policy_decision`. `state.worker_outputs` is accepted as part of
    this node's interface (per the frozen architecture) for future rule
    expansion, but is not consulted by this single rule.

    Args:
        state: The current agent state. May optionally have `task_type`
            and/or `planner_output` set; may optionally have entries in
            `worker_outputs`.

    Returns:
        The same `AgentState` instance, with `selected_critics` and
        `policy_decision` populated.
    """
    task_type = _resolve_task_type(state)
    selected_critics = _select_critics_for_task_type(task_type)

    state.selected_critics = selected_critics
    state.policy_decision = PolicyDecision(
        action="select_critics",
        target_node=NodeName.CRITIC.value,
        rationale=f"Rule-based selection for task_type={task_type!r}: {selected_critics}.",
        metadata={"task_type": task_type, "rule": "task_type_code_else_logic"},
    )
    return state


_CRITIC_REGISTRY: dict[str, type[BaseCritic]] = {
    "LogicCritic": LogicCritic,
    "CodeCritic": CodeCritic,
    "FactCritic": FactCritic,
    "MetaCritic": MetaCritic,
}
"""Maps critic identifiers (as used in `state.selected_critics`) to their `BaseCritic` class."""


def critic_node(state: AgentState) -> AgentState:
    """Execute the selected critics and aggregate their results (Algorithm 1, steps 7-8).

    This is a deterministic placeholder implementation: it performs no
    evaluation logic and makes no LLM calls.

    - Step 7: for each critic identifier in `state.selected_critics`
      (populated by `router_node`), the corresponding `BaseCritic`
      subclass (see `app/critics`) is instantiated and its `evaluate`
      method is called on the latest worker output's text. Every built-in
      critic's `evaluate` is itself a fixed placeholder that ignores its
      input (see `app/critics/critics.py`); unrecognized critic
      identifiers are skipped.
    - Step 8: the resulting `CriticResult`s are combined via
      `MajorityVoteStrategy` (see `app/critics/aggregation.py`), which is
      itself a placeholder that performs no real vote counting.

    Args:
        state: The current agent state. Reads `state.selected_critics` and
            the latest entry of `state.worker_outputs`.

    Returns:
        The same `AgentState` instance, with `state.critic_feedback`
        appended to, `state.critic_scores` updated, and
        `state.aggregated_quality_score` and
        `state.memory_context["critic_aggregation"]` set from the
        (placeholder) aggregation result.
    """
    content = _extract_output_text(_get_latest_worker_output(state))

    results: list[CriticResult] = []
    for critic_name in state.selected_critics:
        critic_class = _CRITIC_REGISTRY.get(critic_name)
        if critic_class is None:
            continue
        results.append(critic_class().evaluate(content))

    aggregated = MajorityVoteStrategy().aggregate(results)

    state.critic_feedback = [
        *state.critic_feedback,
        *[
            CriticFeedback(
                critic_name=result.critic_name,
                feedback=result.feedback or "",
                metadata=result.metadata,
            )
            for result in results
        ],
    ]
    state.critic_scores = {
        **state.critic_scores,
        **{result.critic_name: result.score for result in results},
    }
    state.aggregated_quality_score = aggregated.aggregated_score
    state.memory_context = {
        **state.memory_context,
        "critic_aggregation": aggregated.model_dump(mode="json"),
    }
    return state


def self_correction_node(state: AgentState) -> AgentState:
    """Decide whether to correct via `CorrectionDecisionEngine`, and apply a fixed
    placeholder correction only when that deterministic policy calls for it
    (Algorithm 1, step 9's "if correction required" branch).

    This replaces the previous "always correct" placeholder: reaching this
    node no longer unconditionally applies a correction. Instead:

    1. `CorrectionDecisionEngine.decide` (see `app/correction_policy`)
       evaluates a fixed set of deterministic rules over
       `aggregated_quality_score`, `critic_scores`, `iteration_count`,
       `max_iterations`, `memory_context`, and `error_features` — no
       learning, no LLM calls, no randomness.
    2. The resulting decision is always recorded under
       `state.memory_context["correction_policy"]`, whether or not a
       correction is applied.
    3. If `should_correct` is `False`, the state is returned as-is aside
       from that diagnostic write — no `CorrectionRecord`, no new
       `WorkerOutput`, no `iteration_count` change.
    4. If `should_correct` is `True`, the exact same fixed placeholder
       correction as before is applied: a `CorrectionRecord` is appended,
       `iteration_count` is incremented, and a placeholder "corrected"
       `WorkerOutput` is appended, so a later `evaluation_node` (which
       always reads the latest worker output) naturally returns r*
       instead of r.

    This node still does not decide *whether the graph routes here at
    all* — that remains the not-yet-implemented conditional edges in
    `app/graph/edges.py`; it only decides what to do once reached.

    Args:
        state: The current agent state.

    Returns:
        The same `AgentState` instance, with `state.memory_context`
        updated under the `"correction_policy"` key, and — only if the
        policy calls for it — a new `CorrectionRecord` appended to
        `correction_history`, `iteration_count` incremented, and a new
        `WorkerOutput` appended to `worker_outputs`.
    """
    decision = CorrectionDecisionEngine().decide(state)

    state.memory_context = {
        **state.memory_context,
        "correction_policy": {
            "decision": decision.model_dump(mode="json"),
            "triggered_rules": decision.triggered_rules,
            "confidence": decision.confidence,
            "strategy": decision.decision_strategy,
        },
    }

    if not decision.should_correct:
        return state

    state.correction_history = [
        *state.correction_history,
        CorrectionRecord(
            iteration=state.iteration_count,
            description="Placeholder self-correction: no correction logic implemented.",
            applied_by=NodeName.SELF_CORRECTION.value,
            metadata={},
        ),
    ]
    state.iteration_count += 1

    corrected_output = WorkerOutput(
        worker_id="worker-001",
        worker_name="DefaultWorker",
        worker_type="general",
        input=_resolve_worker_input(state),
        output="Placeholder corrected response.",
        reasoning_summary="No correction reasoning performed.",
        confidence=1.0,
        execution_time=0.0,
        token_usage=0,
        status="corrected",
        metadata={"corrected_at_iteration": state.iteration_count},
    )
    state.worker_outputs = [*state.worker_outputs, corrected_output]
    return state


def safety_node(state: AgentState) -> AgentState:
    """Assess the safety of the current outputs.

    Future responsibility: evaluate the current worker/corrected output
    and update `state.safety_status`.
    """
    raise NotImplementedError("safety_node is a placeholder and is not yet implemented.")


def evaluation_node(state: AgentState) -> AgentState:
    """Finalize the response and store an execution trace (Algorithm 1, step 10).

    This is a deterministic placeholder implementation: it performs no
    evaluation intelligence and makes no LLM calls.

    `state.final_response` is always set from the *latest* entry of
    `state.worker_outputs`. Since both `worker_node` and
    `self_correction_node` append to that same list, this single rule
    transparently returns r* when correction ran (its appended output is
    last) or r otherwise — without this node making any decision itself.

    "Store execution trace" is implemented as a literal, bookkeeping-only
    write to `state.execution_metadata.metadata["trace"]`, reflecting
    counts and identifiers already present in `state` (no computation).

    Once the response is finalized, this node records the completed
    execution as a reusable `ExperienceRecord` (see `app/experience`):
    `ExperienceRecorder` reads only `AgentState` (no business logic, no
    learning, no scoring, no routing).

    Immediately after, `RewardCalculator` (see `app/reward`) converts
    that `ExperienceRecord` into a deterministic `RewardSignal` — again,
    no learning, no policy updates, no repository access of its own.
    Since `ExperienceRecord` is frozen, the reward cannot be attached to
    it in place; instead an enriched *copy* of the record (with the
    reward embedded in `metadata["reward"]`) is what actually gets
    written to `state.memory_context["experience"]` and stored into
    `DEFAULT_EXPERIENCE_REPOSITORY` — the original, reward-less record is
    discarded rather than also stored, so no duplicate ever reaches the
    repository. The reward is additionally written on its own to
    `state.memory_context["reward"]`.

    Finally, `MetricsCollector` (see `app/metrics`) extracts a standardized
    `ExecutionMetrics` from `state`, the enriched `ExperienceRecord`, and
    the `RewardSignal` — pure field extraction, no calculations beyond a
    presence check — written to `state.memory_context["metrics"]` and
    stored into `DEFAULT_METRICS_REPOSITORY`.

    Args:
        state: The current agent state.

    Returns:
        The same `AgentState` instance, with `state.evaluation_metrics`,
        `state.final_response`, `state.execution_metadata`,
        `state.execution_status`, `state.memory_context["experience"]`,
        `state.memory_context["reward"]`, and
        `state.memory_context["metrics"]` populated.
    """
    final_text = _extract_output_text(_get_latest_worker_output(state))

    state.evaluation_metrics = {
        **state.evaluation_metrics,
        "iteration_count": float(state.iteration_count),
        "worker_output_count": float(len(state.worker_outputs)),
        "critic_result_count": float(len(state.critic_scores)),
    }
    state.execution_metadata = ExecutionMetadata(
        created_at=state.execution_metadata.created_at,
        updated_at=datetime.now(timezone.utc),
        metadata={
            **state.execution_metadata.metadata,
            "trace": {
                "planner_ran": state.planner_output is not None,
                "worker_output_count": len(state.worker_outputs),
                "error_feature_count": len(state.error_features),
                "selected_critics": list(state.selected_critics),
                "iteration_count": state.iteration_count,
            },
        },
    )
    state.final_response = final_text
    state.execution_status = ExecutionStatus.COMPLETED

    experience = ExperienceRecorder().record(state)
    reward = RewardCalculator().calculate(experience)
    enriched_experience = experience.model_copy(
        update={"metadata": {**experience.metadata, "reward": reward.model_dump(mode="json")}}
    )
    DEFAULT_EXPERIENCE_REPOSITORY.add(enriched_experience)

    metrics = MetricsCollector(repository=DEFAULT_METRICS_REPOSITORY).collect(
        state, enriched_experience, reward
    )

    state.memory_context = {
        **state.memory_context,
        "experience": enriched_experience.model_dump(mode="json"),
        "reward": reward.model_dump(mode="json"),
        "metrics": metrics.model_dump(mode="json"),
    }
    return state
