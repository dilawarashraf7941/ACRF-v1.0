"""The Experiment Framework: automates reproducible experiments over
`app/evaluation/offline`'s Offline Replay Framework.

- `ExperimentRunner` (`runner.py`) runs `N` independent replays of a
  policy (`HeuristicPolicy` or `LinUCBPolicy` by default) and aggregates
  them into an `ExperimentResult`.
- `Analyzer` (`analyzer.py`) computes mean, standard deviation, min, max,
  95% confidence intervals, and reward/quality/latency trends.
- `Exporter` (`exporter.py`) serializes `ExperimentResult`s to JSON, CSV,
  or Markdown.

No graph integration, no live/online learning, no policy updates during
an experiment, no PPO, and no reinforcement learning are implemented
anywhere in this module. `ReplayEngine`, `OfflineEvaluator`, `Benchmark`,
`ExperienceRepository`, `RewardCalculator`, and every policy are used
as-is, unmodified.
"""

from app.evaluation.experiments.analyzer import DEFAULT_CONFIDENCE_LEVEL, Analyzer
from app.evaluation.experiments.exporter import Exporter
from app.evaluation.experiments.models import (
    ConfidenceInterval,
    ExperimentConfig,
    ExperimentResult,
    StatisticalSummary,
)
from app.evaluation.experiments.runner import (
    BootstrapExperienceRepository,
    ExperimentRunner,
)

__all__ = [
    "DEFAULT_CONFIDENCE_LEVEL",
    "Analyzer",
    "BootstrapExperienceRepository",
    "ConfidenceInterval",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentRunner",
    "Exporter",
    "StatisticalSummary",
]
