"""The Ablation Study Framework: systematically compares system variants
using the existing evaluation pipeline.

- `AblationRunner` (`runner.py`) runs one of seven supported ablation
  recipes by composing `app.evaluation.experiments.ExperimentRunner`
  (itself composing `app.evaluation.offline.ReplayEngine`/
  `OfflineEvaluator`), `app.evaluation.offline.Benchmark`, and
  `app.evaluation.statistics.Analyzer` — no replay, resampling, reward,
  or statistical-test logic is reimplemented here.
- `AblationReportGenerator` (`report.py`) formats a list of
  `AblationResult`s as Markdown (with a summary table, ranking, best/worst
  configuration, and key observations), CSV, or JSON.

No reinforcement learning, no PPO, and no learning of any kind.
`app/graph`, `app/router`, `app/policy_engine`,
`app/evaluation/offline/replay.py` (`ReplayEngine`),
`app/evaluation/experiments` (`ExperimentRunner`), `app/reward`,
`app/experience`, and `app/context` are all used as-is, unmodified.
"""

from app.evaluation.ablation.models import AblationConfig, AblationResult
from app.evaluation.ablation.report import AblationReportGenerator
from app.evaluation.ablation.runner import (
    DEFAULT_ALPHA_SWEEP,
    DEFAULT_KEEP_FEATURE_FRACTION,
    AblationRunner,
    QualityOnlyRewardStrategy,
    RandomCriticPolicy,
    ReducedContextPolicy,
)

__all__ = [
    "DEFAULT_ALPHA_SWEEP",
    "DEFAULT_KEEP_FEATURE_FRACTION",
    "AblationConfig",
    "AblationReportGenerator",
    "AblationResult",
    "AblationRunner",
    "QualityOnlyRewardStrategy",
    "RandomCriticPolicy",
    "ReducedContextPolicy",
]
