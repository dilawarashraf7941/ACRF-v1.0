"""Unit tests for the deterministic, heuristic-only `error_feature_extractor_node`."""

from app.graph.nodes import NodeName, error_feature_extractor_node
from app.state import AgentState, ExecutionStatus, SafetyStatus, WorkerOutput


def _make_state(task_type: str | None = None) -> AgentState:
    return AgentState(session_id="session-1", task_id="task-1", user_query="q", task_type=task_type)


def _with_worker_output(state: AgentState, output: str) -> AgentState:
    state.worker_outputs = [WorkerOutput(worker_id="worker-001", output=output)]
    return state


# --- Empty output ---


def test_empty_output_yields_zero_confidence_high_risk() -> None:
    state = _with_worker_output(_make_state(), "   ")

    result = error_feature_extractor_node(state)
    features = result.error_features[-1]

    assert features.metadata["confidence"] == 0.0
    assert features.metadata["risk_level"] == "high"
    assert features.metadata["error_category"] == "empty_output"


def test_truly_empty_string_output_is_treated_as_empty() -> None:
    state = _with_worker_output(_make_state(), "")

    result = error_feature_extractor_node(state)

    assert result.error_features[-1].metadata["error_category"] == "empty_output"


def test_no_worker_outputs_at_all_is_treated_as_empty() -> None:
    state = _make_state()
    assert state.worker_outputs == []

    result = error_feature_extractor_node(state)

    assert result.error_features[-1].metadata["error_category"] == "empty_output"
    assert result.error_features[-1].metadata["confidence"] == 0.0


# --- Short output ---


def test_short_output_yields_partial_confidence_medium_risk() -> None:
    state = _with_worker_output(_make_state(), "short")

    result = error_feature_extractor_node(state)
    features = result.error_features[-1]

    assert features.metadata["confidence"] == 0.4
    assert features.metadata["risk_level"] == "medium"
    assert features.metadata["error_category"] == "short_output"


def test_short_output_boundary_just_under_threshold_is_short() -> None:
    text = "x" * 19  # < 20 chars
    state = _with_worker_output(_make_state(), text)

    result = error_feature_extractor_node(state)

    assert result.error_features[-1].metadata["error_category"] == "short_output"


def test_output_at_threshold_length_is_not_short() -> None:
    text = "x" * 20  # == 20 chars, not < 20
    state = _with_worker_output(_make_state(), text)

    result = error_feature_extractor_node(state)

    assert result.error_features[-1].metadata["error_category"] == "none"


# --- Normal (non-empty, non-short) output ---


def test_normal_length_output_yields_full_confidence_low_risk() -> None:
    text = "This is a perfectly normal length plain text answer with no code at all."
    state = _with_worker_output(_make_state(), text)

    result = error_feature_extractor_node(state)
    features = result.error_features[-1]

    assert features.metadata["confidence"] == 1.0
    assert features.metadata["risk_level"] == "low"
    assert features.metadata["error_category"] == "none"


# --- Code detection ---


def test_code_keyword_is_detected_as_code() -> None:
    text = "def compute_total(items): return sum(items)"
    state = _with_worker_output(_make_state(), text)

    result = error_feature_extractor_node(state)
    features = result.error_features[-1]

    assert features.metadata["output_type"] == "code"
    assert features.metadata["suggested_critics"] == ["CodeCritic"]


def test_code_keyword_detection_is_case_insensitive() -> None:
    text = "RETURN the value after processing it fully please"
    state = _with_worker_output(_make_state(), text)

    result = error_feature_extractor_node(state)

    assert result.error_features[-1].metadata["output_type"] == "code"


def test_fenced_code_block_is_detected_as_code() -> None:
    text = "Here is the answer:\n```\nprint('hello world')\n```\nThat is the full result."
    state = _with_worker_output(_make_state(), text)

    result = error_feature_extractor_node(state)

    assert result.error_features[-1].metadata["output_type"] == "code"
    assert result.error_features[-1].metadata["suggested_critics"] == ["CodeCritic"]


def test_plain_text_without_keywords_is_not_detected_as_code() -> None:
    text = "The weather today is nice outside and everyone seems quite happy about it."
    state = _with_worker_output(_make_state(), text)

    result = error_feature_extractor_node(state)
    features = result.error_features[-1]

    assert features.metadata["output_type"] == "text"
    assert features.metadata["suggested_critics"] == ["LogicCritic"]


def test_short_code_output_still_detected_as_code() -> None:
    # Short (<20 chars) AND contains a code keyword: code detection and
    # empty/short-length classification are independent heuristics.
    state = _with_worker_output(_make_state(), "return 1")

    result = error_feature_extractor_node(state)
    features = result.error_features[-1]

    assert features.metadata["error_category"] == "short_output"
    assert features.metadata["output_type"] == "code"
    assert features.metadata["suggested_critics"] == ["CodeCritic"]


# --- task_type resolution ---


def test_task_type_uses_state_task_type() -> None:
    state = _with_worker_output(_make_state(task_type="code"), "plain text output here")

    result = error_feature_extractor_node(state)

    assert result.error_features[-1].metadata["task_type"] == "code"


def test_task_type_is_none_when_unresolvable() -> None:
    state = _with_worker_output(_make_state(task_type=None), "plain text output here")

    result = error_feature_extractor_node(state)

    assert result.error_features[-1].metadata["task_type"] is None


# --- ErrorFeatureProfile / ErrorFeatureCollection fidelity ---


def test_metadata_contains_full_profile_and_collection_dumps() -> None:
    state = _with_worker_output(_make_state(task_type="code"), "def compute(x): return x + 1")

    result = error_feature_extractor_node(state)
    metadata = result.error_features[-1].metadata

    assert metadata["profile"]["risk_level"] == "low"
    assert metadata["profile"]["suggested_critics"] == ["CodeCritic"]
    assert metadata["profile"]["task_type"] == "code"
    assert metadata["profile"]["output_type"] == "code"
    assert metadata["profile"]["error_category"] == "none"
    assert len(metadata["collection"]["features"]) == 1
    assert metadata["collection"]["overall_risk_level"] == "low"


def test_bridged_error_feature_matches_frozen_error_feature_shape() -> None:
    state = _with_worker_output(_make_state(), "   ")

    result = error_feature_extractor_node(state)
    feature = result.error_features[-1]

    assert feature.error_type == "empty_output"
    assert feature.severity == "high"
    assert feature.source_node == NodeName.ERROR_FEATURE_EXTRACTOR.value
    assert isinstance(feature.description, str) and feature.description


# --- General node behavior ---


def test_appends_without_discarding_existing_error_features() -> None:
    state = _with_worker_output(_make_state(), "   ")

    error_feature_extractor_node(state)
    error_feature_extractor_node(state)

    assert len(state.error_features) == 2


def test_is_deterministic() -> None:
    text = "def f(): return 1"

    result_a = error_feature_extractor_node(_with_worker_output(_make_state(task_type="code"), text))
    result_b = error_feature_extractor_node(_with_worker_output(_make_state(task_type="code"), text))

    metadata_a = dict(result_a.error_features[-1].metadata)
    metadata_b = dict(result_b.error_features[-1].metadata)
    # extraction_metadata.extracted_at is a real timestamp; strip it for comparison.
    metadata_a["profile"].pop("extraction_metadata", None)
    metadata_b["profile"].pop("extraction_metadata", None)
    metadata_a["collection"]["features"][0].pop("extraction_metadata", None)
    metadata_b["collection"]["features"][0].pop("extraction_metadata", None)

    assert metadata_a == metadata_b


def test_returns_same_state_instance() -> None:
    state = _with_worker_output(_make_state(), "text")

    result = error_feature_extractor_node(state)

    assert result is state


def test_does_not_modify_unrelated_state_fields() -> None:
    state = _with_worker_output(_make_state(), "text")

    result = error_feature_extractor_node(state)

    assert result.session_id == "session-1"
    assert result.task_id == "task-1"
    assert result.selected_critics == []
    assert result.policy_decision is None
    assert result.iteration_count == 0
    assert result.max_iterations == 10
    assert result.safety_status == SafetyStatus.UNKNOWN
    assert result.execution_status == ExecutionStatus.PENDING
    assert result.final_response is None


def test_uses_content_field_when_output_field_is_absent() -> None:
    state = _make_state()
    state.worker_outputs = [WorkerOutput(worker_id="worker-001", content="def f(): return 1")]

    result = error_feature_extractor_node(state)

    assert result.error_features[-1].metadata["output_type"] == "code"
