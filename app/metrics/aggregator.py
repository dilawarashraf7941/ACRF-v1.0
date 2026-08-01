"""`MetricsAggregator`: converts a collection of `ExecutionMetrics` into an
`ExperimentSummary`.

Every statistic here is a simple, deterministic arithmetic mean, rate, or
count — no reinforcement learning, no contextual bandits, no policy
optimization, and no randomness. Empty input is handled gracefully:
every average/rate is `None` rather than raising or reporting a
misleading `0.0`.
"""

from collections import Counter

from app.metrics.models import ExecutionMetrics, ExperimentSummary

UNKNOWN_POLICY_LABEL = "unknown"
"""Grouping key used for a metrics record whose `metadata["policy"]` is missing or not a string."""


def _policy_of(metrics: ExecutionMetrics) -> str:
    """Read the grouping policy tag off a single `ExecutionMetrics`.

    Args:
        metrics: The record to read.

    Returns:
        `metrics.metadata["policy"]` if present and a string, else
        `UNKNOWN_POLICY_LABEL`.
    """
    policy = metrics.metadata.get("policy")
    return policy if isinstance(policy, str) else UNKNOWN_POLICY_LABEL


def _mean(values: list[float]) -> float | None:
    """Compute the arithmetic mean of `values`, or `None` if empty.

    Args:
        values: The values to average.

    Returns:
        `sum(values) / len(values)`, or `None` if `values` is empty.
    """
    if not values:
        return None
    return sum(values) / len(values)


def _mean_non_empty(values: list[float]) -> float:
    """Compute the arithmetic mean of a `values` list known to be non-empty.

    Args:
        values: A non-empty list of values to average.

    Returns:
        `sum(values) / len(values)`.
    """
    return sum(values) / len(values)


class MetricsAggregator:
    """Deterministically aggregates a list of `ExecutionMetrics` into an `ExperimentSummary`."""

    def aggregate(self, metrics: list[ExecutionMetrics]) -> ExperimentSummary:
        """Compute an `ExperimentSummary` over `metrics`.

        Args:
            metrics: The `ExecutionMetrics` records to summarize. May be empty.

        Returns:
            An `ExperimentSummary` with `total_runs = len(metrics)`, every
            average/rate set to `None` when `metrics` is empty (or, for a
            given field, when every record's value for it was `None`),
            and every dict statistic set to `{}` when `metrics` is empty.
        """
        total_runs = len(metrics)

        rewards = [m.reward for m in metrics]
        quality_scores = [
            m.aggregated_quality_score for m in metrics if m.aggregated_quality_score is not None
        ]
        iteration_counts = [float(m.iterations) for m in metrics]
        latencies = [m.latency for m in metrics if m.latency is not None]
        costs = [m.estimated_cost for m in metrics if m.estimated_cost is not None]

        success_rate = (
            sum(1 for m in metrics if m.execution_status == "completed") / total_runs
            if total_runs
            else None
        )
        correction_rate = (
            sum(1 for m in metrics if m.correction_applied) / total_runs if total_runs else None
        )

        policy_usage = Counter(_policy_of(m) for m in metrics)

        reward_totals_by_policy: dict[str, list[float]] = {}
        for m in metrics:
            reward_totals_by_policy.setdefault(_policy_of(m), []).append(m.reward)
        # Every group is non-empty by construction (each was seeded by
        # appending at least one reward).
        average_reward_per_policy = {
            policy: _mean_non_empty(policy_rewards)
            for policy, policy_rewards in reward_totals_by_policy.items()
        }

        critic_selection_frequency = Counter(
            critic_name for m in metrics for critic_name in m.selected_critics
        )

        return ExperimentSummary(
            total_runs=total_runs,
            average_reward=_mean(rewards),
            average_quality=_mean(quality_scores),
            average_iterations=_mean(iteration_counts),
            average_latency=_mean(latencies),
            average_cost=_mean(costs),
            success_rate=success_rate,
            correction_rate=correction_rate,
            average_reward_per_policy={
                policy: round(value, 6) for policy, value in average_reward_per_policy.items()
            },
            critic_selection_frequency=dict(critic_selection_frequency),
            policy_usage=dict(policy_usage),
            metadata={
                "runs_with_quality_score": len(quality_scores),
                "runs_with_latency": len(latencies),
                "runs_with_cost": len(costs),
            },
        )
