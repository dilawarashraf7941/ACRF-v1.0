"""`ReplayEngine`: replays stored `ExperienceRecord`s against one policy, offline.

Implements the "replay method" for offline (off-policy) evaluation of
logged bandit feedback (Li et al., 2011, "Unbiased Offline Evaluation of
Contextual-bandit-based News Article Recommendation Algorithms"): for
each stored experience, the policy being replayed selects an action from
a fixed candidate set given a `ContextVector` built from that experience
alone. If the policy's selection exactly matches the critic(s) actually
recorded for that experience, the experience's already-recorded reward is
*replayed* for that matched decision (computed via
`RewardCalculator.calculate`, never fabricated or estimated). If the
policy would have chosen differently, the experience is skipped for this
policy's evaluation — its outcome under that untaken action is unknown.

Two distinct properties are easy to conflate here, so they are stated
separately. (1) Reward-attribution correctness for the matched subset:
whenever the policy's action exactly matches the logged action, replaying
that experience's already-recorded reward is exact, not estimated — this
holds unconditionally, for any context. (2) Estimator unbiasedness in the
formal Li et al. sense: their proof additionally assumes the *logging*
policy selected actions with known, non-degenerate probability (uniform
random, in their deployment), so that every candidate action had a chance
of being logged for a given context. ACRF's logged actions come from
`router_node`'s fixed, deterministic, task-type-based rule, not a
randomized exploration policy — so this module does not claim property
(2); no inverse-propensity weighting is computed, and a `ReplayEngine`
result should be read as an exact replay of matched historical decisions,
not as a statistically unbiased estimate of a candidate policy's
performance across the full action space. See `README.md` in this
package for the full discussion.

Separately, and independent of the estimator-unbiasedness question above:
the `ContextVector` a policy selects against must itself contain only
information that would genuinely be available *before* a live routing
decision is made. `build_offline_context_vector` (below) is restricted to
`ExperienceRecord` fields verified invariant across a whole recorded
episode for exactly this reason — see that function's docstring.

No LLM call, no graph execution, and no new `ExperienceRecord` is ever
created or stored. `ReplayEngine.replay` (the original, deterministic
mode) only ever calls `repository.list()` (read-only) and
`policy.select_action(...)` — it never calls an `update` method on the
policy, so replaying never performs live/online learning, even for a
policy (like `LinUCBPolicy`) that has one. This mode, and every line of
its implementation, is unchanged from its original form.

`ReplayEngine.replay_with_learning` is an additional, explicitly opt-in
mode: it processes the same matched experiences sequentially and, after
computing each one's reward, calls `policy.update(context, action,
reward)` before moving to the next experience — so the policy's state
evolves across the pass. It exists specifically so the two modes can be
run side by side and compared (same repository, same starting policy
state, same candidate actions): `replay` measures a policy exactly as
constructed; `replay_with_learning` measures how that same starting
policy performs once it has learned, sequentially, from the log.
"""

import hashlib
from typing import Protocol, runtime_checkable

from app.context import ContextVector
from app.evaluation.offline.models import ReplayStep
from app.experience import ExperienceRecord, ExperienceRepository
from app.reward import RewardCalculator

DEFAULT_CANDIDATE_CRITICS: tuple[str, ...] = (
    "LogicCritic",
    "CodeCritic",
    "FactCritic",
    "MetaCritic",
)
"""The fixed, built-in critic identifiers replayed against by default.

Mirrors `app/graph/nodes.py`'s `_CANDIDATE_CRITIC_NAMES` (the live
candidate set `policy_engine_node` scores), duplicated here rather than
imported so `app/evaluation/offline` has no dependency on `app/graph` —
this module must remain usable standalone, entirely offline.
"""


@runtime_checkable
class ReplayablePolicy(Protocol):
    """Structural interface for any policy `ReplayEngine` can replay.

    Matches both `app.policy.base.BasePolicy`
    (`select_action(context, candidate_critics) -> PolicyDecision`) and
    `app.policy.linucb.LinUCBPolicy`
    (`select_action(context, actions) -> LinUCBSelection`) — and any
    future Contextual Bandit, Offline RL, or Thompson Sampling policy —
    without either implementing this `Protocol` explicitly (structural,
    not nominal, typing), and without this module importing from
    `app.policy` or `app.policy.linucb` at all. Only the shape of the
    call matters: two positional arguments, `context` and a list of
    candidate action identifiers, returning any object.
    """

    def select_action(self, context: ContextVector, candidate_actions: list[str]) -> object:
        """Select an action for `context` from `candidate_actions`."""
        ...


@runtime_checkable
class TrainablePolicy(ReplayablePolicy, Protocol):
    """Structural interface for a policy `ReplayEngine.replay_with_learning` can train.

    Everything `ReplayablePolicy` requires, plus `update(context, action,
    reward)` — the exact signature `app.policy.linucb.LinUCBPolicy.update`
    already has. `HeuristicPolicy` and other stateless policies do not
    satisfy this Protocol (they have no `update` method), and
    `replay_with_learning` raises immediately if asked to train one.
    """

    def update(self, context: ContextVector, action: str, reward: float) -> None:
        """Incorporate one `(context, action, reward)` observation."""
        ...


def _build_offline_context_id(experience_id: str) -> str:
    """Deterministically derive a context id for an offline-replay `ContextVector`.

    Salted with `"offline_replay|"` so this id can never collide with a
    live `ContextEncoder`-built context id (see
    `app/context/encoder.py::_build_context_id`), even for the same
    underlying experience.

    Args:
        experience_id: `ExperienceRecord.experience_id`.

    Returns:
        A 64-character hex digest identifying this experience's offline-replay context.
    """
    seed = f"offline_replay|{experience_id}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


_PLAN_COMPLEXITY_STEP_CAP = 5.0
"""Mirrors `app/context/encoder.py`'s `_PLAN_COMPLEXITY_STEP_CAP`.

Duplicated rather than imported, consistent with this module's existing,
deliberate independence from `app/context` (see `build_offline_context_vector`'s
docstring). Keeping the same constant value means a plan-complexity score
built here is comparable in scale to the live `ContextEncoder`'s, even
though the two are computed by separate code paths.
"""

_MAX_ITERATIONS_BOUNDS: tuple[float, float] = (1.0, 20.0)
"""Mirrors `app/context/normalizer.py`'s `FEATURE_BOUNDS["max_iterations"]`, for the same reason."""


def _clamp(value: float) -> float:
    """Clamp `value` to `[0.0, 1.0]`."""
    return max(0.0, min(1.0, value))


def _extract_plan_complexity(planner_output: object) -> float:
    """Score plan complexity from a `state_features["planner_output"]` dump, or `0.0`.

    Reads `planner_output["decomposition"]` — the same field
    `app/context/encoder.py::_extract_plan_complexity` reads off a live
    `PlannerOutput` — and applies the same `len(decomposition) /
    _PLAN_COMPLEXITY_STEP_CAP`, clamped formula, so a plan built once by
    `planner_node` scores identically whether read live or from a stored
    experience. Returns `0.0` for anything not shaped as expected
    (including `None`, since no planner ran, or the current placeholder
    planner, which always produces an empty decomposition).
    """
    if not isinstance(planner_output, dict):
        return 0.0
    decomposition = planner_output.get("decomposition")
    if not isinstance(decomposition, list):
        return 0.0
    return _clamp(len(decomposition) / _PLAN_COMPLEXITY_STEP_CAP)


def _normalize_max_iterations(value: float) -> float:
    """Min-max-scale `value` against `_MAX_ITERATIONS_BOUNDS`, mirroring `FeatureNormalizer`."""
    low, high = _MAX_ITERATIONS_BOUNDS
    if high <= low:
        return value
    return _clamp((value - low) / (high - low))


def build_offline_context_vector(experience: ExperienceRecord) -> ContextVector:
    """Build a `ContextVector` from a stored `ExperienceRecord` alone.

    `app.context.ContextEncoder` cannot be reused here: it only encodes a
    live `AgentState` (`ContextEncoder.encode(state, ...)`), and a stored
    `ExperienceRecord` is not one — offline replay, by definition, has no
    live `AgentState` to encode, only the distilled record a past
    execution left behind. This function is therefore a separate,
    self-contained context builder, deliberately independent of
    `app/context` (the same "small modules stay independent" convention
    `app/correction_policy` and `app/context` itself already follow) —
    `app/context/encoder.py` is not modified, imported from, or
    duplicated here beyond producing the same `ContextVector` output type.

    Feature selection is deliberately restricted to fields verified
    invariant across a whole recorded episode, and therefore safe to treat
    as available *before* a routing decision:

    - `is_code_task` / `has_task_type`: from `state_features["task_type"]`,
      which is `AgentState.task_type` — set once at state construction,
      before the graph runs, and never reassigned by any node.
    - `plan_complexity`: from `state_features["planner_output"]`, which is
      `AgentState.planner_output` — assigned exactly once, by
      `planner_node`, the graph's first node; `self_correction_node` does
      not re-invoke the planner or otherwise mutate it.
    - `max_iterations`: from `state_features["max_iterations"]`, a fixed
      configuration/budget value never reassigned by any node.

    `ExperienceRecord` is built exactly once, at the *end* of an episode
    (`evaluation_node`, via `ExperienceRecorder`) — there is no separate,
    pre-decision snapshot in the current data model. Every other
    `ExperienceRecord` field (`aggregated_quality_score`, `iterations`,
    `latency`, `estimated_cost`, `critic_scores`, `selected_critics`,
    `correction_decision`, `execution_status`, and the
    `state_features["error_feature_count"]` /
    `state_features["worker_output_count"]` counts) reflects the
    episode's outcome, the action actually taken, or a value that can
    change across the episode's iterations (`error_feature_extractor_node`
    re-runs on every `worker -> error_feature_extractor` pass) — using any
    of them here would let the policy's input be informed by information
    that would not genuinely be available at decision time (target/outcome
    leakage). They are excluded for that reason, not by oversight; see
    `README.md` in this package for the full audit. A prior version of
    this function used those fields; this is an intentional, breaking
    change to the *content* of the returned `ContextVector.features`
    (the function's signature and return type are unchanged).

    Every feature is a plain, deterministic transcription or ratio of an
    already-recorded `ExperienceRecord` field; nothing is fitted, learned,
    or randomized, and `experience` itself is never modified.

    Args:
        experience: The stored experience to build a context from.

    Returns:
        A `ContextVector` with `source_execution_id=experience.experience_id`
        and `timestamp=experience.timestamp` (never the current wall-clock
        time, so this function is a pure, deterministic function of its input).
    """
    state_features = (
        experience.state_features if isinstance(experience.state_features, dict) else {}
    )
    task_type = state_features.get("task_type")
    max_iterations = state_features.get("max_iterations")

    features: dict[str, float] = {
        "is_code_task": 1.0 if task_type == "code" else 0.0,
        "has_task_type": 1.0 if task_type else 0.0,
        "plan_complexity": _extract_plan_complexity(state_features.get("planner_output")),
        "max_iterations": (
            _normalize_max_iterations(float(max_iterations))
            if isinstance(max_iterations, (int, float))
            else 0.0
        ),
    }

    return ContextVector(
        context_id=_build_offline_context_id(experience.experience_id),
        source_execution_id=experience.experience_id,
        features=features,
        feature_order=list(features.keys()),
        timestamp=experience.timestamp,
        metadata={"source": "OfflineReplay", "experience_id": experience.experience_id},
    )


def _extract_selected_critics(decision: object) -> list[str]:
    """Normalize any policy decision object into a list of selected critic identifiers.

    Supports any decision object exposing either `selected_critics: list[str]`
    (e.g. `app.policy.models.PolicyDecision`) or `selected_action: str`
    (e.g. `app.policy.linucb.models.LinUCBSelection`) — covering every
    current policy, and any future one following either convention,
    without this module importing either decision type.

    Args:
        decision: The object returned by `ReplayablePolicy.select_action`.

    Returns:
        The selected critic identifier(s) as a list (a single-item list
        for a `selected_action`-style decision).

    Raises:
        TypeError: If `decision` exposes neither attribute.
    """
    selected_critics = getattr(decision, "selected_critics", None)
    if selected_critics is not None:
        return list(selected_critics)

    selected_action = getattr(decision, "selected_action", None)
    if selected_action is not None:
        return [selected_action]

    raise TypeError(
        f"{type(decision).__name__} exposes neither 'selected_critics' nor "
        "'selected_action'; ReplayEngine cannot determine which action(s) were selected."
    )


class ReplayEngine:
    """Replays every stored `ExperienceRecord` against one policy, deterministically.

    Constructed via dependency injection with the three collaborators
    named in this framework's spec: an `ExperienceRepository` to read
    from, a `ReplayablePolicy` to replay, and a `RewardCalculator` to
    replay each matched experience's reward with. Nothing about this
    class is specific to `HeuristicPolicy` or `LinUCBPolicy` — any object
    implementing `ReplayablePolicy` works unmodified, including a future
    `OfflineRLPolicy` or `ThompsonSamplingPolicy`.
    """

    def __init__(
        self,
        repository: ExperienceRepository,
        policy: ReplayablePolicy,
        reward_calculator: RewardCalculator,
        candidate_actions: list[str] | None = None,
    ) -> None:
        """Create an engine wired to `repository`, `policy`, and `reward_calculator`.

        Args:
            repository: Where stored `ExperienceRecord`s are read from.
                Never written to by this class.
            policy: The policy being replayed. Only `select_action` is
                ever called on it — never an `update` method, even if one
                exists — so replay never performs live/online learning.
            reward_calculator: Computes each matched experience's reward.
            candidate_actions: The fixed candidate action set every
                experience is replayed against. Defaults to
                `DEFAULT_CANDIDATE_CRITICS`.
        """
        self._repository = repository
        self._policy = policy
        self._reward_calculator = reward_calculator
        self._candidate_actions = (
            list(candidate_actions)
            if candidate_actions is not None
            else list(DEFAULT_CANDIDATE_CRITICS)
        )

    @property
    def repository(self) -> ExperienceRepository:
        """The `ExperienceRepository` this engine replays from."""
        return self._repository

    @property
    def candidate_actions(self) -> list[str]:
        """A copy of the fixed candidate action set every experience is replayed against."""
        return list(self._candidate_actions)

    def replay(self) -> list[ReplayStep]:
        """Replay every stored experience, returning one `ReplayStep` per match.

        For each `ExperienceRecord` in `repository.list()` (read-only):

        1. Build a `ContextVector` via `build_offline_context_vector`.
        2. Call `policy.select_action(context, candidate_actions)`.
        3. If the policy's selection (as a set) exactly equals
           `experience.selected_critics` (as a set), replay this
           experience's reward via `reward_calculator.calculate` and
           record a `ReplayStep`. Otherwise, skip it.

        Returns:
            One `ReplayStep` per matched experience, in `repository.list()`'s
            order (filtered). `experience` is never mutated; no new
            `ExperienceRecord` is created or stored.
        """
        steps: list[ReplayStep] = []
        for experience in self._repository.list():
            context = build_offline_context_vector(experience)
            decision = self._policy.select_action(context, list(self._candidate_actions))
            policy_selected = _extract_selected_critics(decision)

            if not policy_selected:
                continue
            if set(policy_selected) != set(experience.selected_critics):
                continue

            reward_signal = self._reward_calculator.calculate(experience)
            steps.append(
                ReplayStep(
                    experience_id=experience.experience_id,
                    context_id=context.context_id,
                    selected_critics=policy_selected,
                    reward=reward_signal.reward,
                    quality=experience.aggregated_quality_score,
                    iterations=experience.iterations,
                    latency=experience.latency,
                )
            )
        return steps

    def replay_with_learning(self) -> list[ReplayStep]:
        """Sequentially replay every stored experience, training the policy as it goes.

        The **optional, explicitly opt-in** sequential learning mode:
        for each `ExperienceRecord` in `repository.list()`'s order,
        in sequence —

        1. Build a `ContextVector` via `build_offline_context_vector`.
        2. Call `policy.select_action(context, candidate_actions)`
           ("replay experience").
        3. If the policy's selection (as a set) does not exactly equal
           `experience.selected_critics` (as a set), skip this
           experience — exactly like `replay`, this mode never
           fabricates a reward for an action that was never actually
           taken; it can only train on what the log actually observed.
        4. Otherwise, compute this experience's reward via
           `reward_calculator.calculate` ("receive reward") and
           immediately call `policy.update(context, critic, reward)`
           for every selected critic ("call policy.update()") — *before*
           moving to the next experience ("continue to next
           experience"), so a later experience in this same call is
           scored by whatever the policy has already learned from
           earlier ones.

        This is the only method on `ReplayEngine` that ever calls
        `update` on the policy; `replay` (above) is completely
        unaffected by this method's existence and never trains anything,
        so the two can be run back to back on freshly-constructed,
        identically-seeded policies to compare a policy's out-of-the-box
        behavior against its behavior after sequential training on the
        same log.

        Returns:
            One `ReplayStep` per matched experience, in the same order
            and shape `replay` returns — directly comparable to a
            `replay()` call's output. `experience` is never mutated; no
            new `ExperienceRecord` is created or stored. Only `policy`'s
            own internal state changes (via `update`).

        Raises:
            AttributeError: If this engine's policy has no `update`
                method (e.g. `HeuristicPolicy`) — checked once, up
                front, so a non-trainable policy fails fast with a clear
                message rather than partway through the pass.
        """
        if not hasattr(self._policy, "update"):
            raise AttributeError(
                f"{type(self._policy).__name__} has no update() method; "
                "replay_with_learning requires a trainable policy (e.g. LinUCBPolicy)."
            )

        steps: list[ReplayStep] = []
        for experience in self._repository.list():
            context = build_offline_context_vector(experience)
            decision = self._policy.select_action(context, list(self._candidate_actions))
            policy_selected = _extract_selected_critics(decision)

            if not policy_selected:
                continue
            if set(policy_selected) != set(experience.selected_critics):
                continue

            reward_signal = self._reward_calculator.calculate(experience)
            for critic_name in policy_selected:
                self._policy.update(context, critic_name, reward_signal.reward)

            steps.append(
                ReplayStep(
                    experience_id=experience.experience_id,
                    context_id=context.context_id,
                    selected_critics=policy_selected,
                    reward=reward_signal.reward,
                    quality=experience.aggregated_quality_score,
                    iterations=experience.iterations,
                    latency=experience.latency,
                )
            )
        return steps
