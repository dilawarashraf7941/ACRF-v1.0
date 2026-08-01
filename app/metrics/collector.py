"""`MetricsCollector`: builds `ExecutionMetrics` from a completed execution's
`AgentState`, `ExperienceRecord`, and `RewardSignal`.

Performs no calculations other than extracting values — no learning, no
policy optimization, no scoring, and no routing. Every field on the
resulting `ExecutionMetrics` is either copied directly from one of the
three inputs, or a trivial presence/length check (never arithmetic).
"""

from app.experience import ExperienceRecord
from app.metrics.models import ExecutionMetrics
from app.metrics.repository import MetricsRepository
from app.reward import RewardSignal
from app.state import AgentState

DEFAULT_POLICY_NAME = "HeuristicPolicy"
"""The policy tag used when no explicit policy identifier is published into
`state.memory_context`. Currently the only policy implementation ACRF has is
the deterministic Heuristic Policy (see `app/policy_engine`); a future
Contextual Bandit / Offline RL / PPO / Q-learning policy node can publish
its own name to `state.memory_context["policy_engine"]["policy_name"]` and
`MetricsCollector` will pick it up automatically, with no change to this
module."""


def _extract_policy_name(state: AgentState) -> str:
    """Read the active policy's identifier from `state.memory_context`, if published.

    Args:
        state: The current agent state.

    Returns:
        `state.memory_context["policy_engine"]["policy_name"]` if present
        and a string, else `DEFAULT_POLICY_NAME`.
    """
    policy_engine_entry = state.memory_context.get("policy_engine")
    if isinstance(policy_engine_entry, dict):
        policy_name = policy_engine_entry.get("policy_name")
        if isinstance(policy_name, str):
            return policy_name
    return DEFAULT_POLICY_NAME


class MetricsCollector:
    """Converts `(AgentState, ExperienceRecord, RewardSignal)` into an `ExecutionMetrics`.

    Optionally accepts a `MetricsRepository` (dependency injection), in
    which case every record built by `collect` is also stored into it.
    Without a repository, `collect` only builds and returns the record.
    """

    def __init__(self, repository: MetricsRepository | None = None) -> None:
        """Create a collector, optionally wired to a repository.

        Args:
            repository: A `MetricsRepository` to store every collected
                metrics record into, or `None` to only build records
                without storing them.
        """
        self._repository = repository

    def collect(
        self, state: AgentState, experience: ExperienceRecord, reward: RewardSignal
    ) -> ExecutionMetrics:
        """Build an `ExecutionMetrics` from a completed execution's state, experience, and reward.

        Args:
            state: The current agent state. Read-only: no field of
                `state` is modified.
            experience: The `ExperienceRecord` built for this execution.
            reward: The `RewardSignal` computed for this execution.

        Returns:
            The newly built `ExecutionMetrics`.
        """
        metrics = ExecutionMetrics(
            execution_id=experience.experience_id,
            reward=reward.reward,
            aggregated_quality_score=experience.aggregated_quality_score,
            iterations=experience.iterations,
            latency=experience.latency,
            estimated_cost=experience.estimated_cost,
            selected_critics=list(experience.selected_critics),
            correction_applied=len(state.correction_history) > 0,
            execution_status=experience.execution_status,
            timestamp=experience.timestamp,
            metadata={
                "policy": _extract_policy_name(state),
                "reward_strategy": reward.strategy,
                "session_id": experience.session_id,
                "task_id": experience.task_id,
            },
        )

        if self._repository is not None:
            self._repository.add(metrics)

        return metrics
