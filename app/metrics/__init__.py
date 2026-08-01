"""The Metrics & Experiment Framework: collects standardized evaluation
metrics for research experiments.

No reinforcement learning, no contextual bandits, no policy optimization,
and no LLM calls — this module only extracts and aggregates already-
computed values from `AgentState`, `ExperienceRecord`, and `RewardSignal`.
Must work unchanged for the current Heuristic Policy and for any future
Contextual Bandit / Offline RL / PPO / Q-learning policy.
"""

from app.metrics.aggregator import MetricsAggregator
from app.metrics.collector import DEFAULT_POLICY_NAME, MetricsCollector
from app.metrics.models import ExecutionMetrics, ExperimentSummary
from app.metrics.repository import (
    DEFAULT_METRICS_REPOSITORY,
    InMemoryMetricsRepository,
    MetricsRepository,
)

__all__ = [
    "DEFAULT_METRICS_REPOSITORY",
    "DEFAULT_POLICY_NAME",
    "ExecutionMetrics",
    "ExperimentSummary",
    "InMemoryMetricsRepository",
    "MetricsAggregator",
    "MetricsCollector",
    "MetricsRepository",
]
