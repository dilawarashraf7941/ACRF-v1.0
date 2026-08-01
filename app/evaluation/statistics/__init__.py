"""The Statistical Analysis Framework: compares experimental results scientifically.

- `Analyzer` (`analyzer.py`) runs a paired statistical comparison between
  two policies' `ExperimentResult`s (or plain numeric sequences):
  automatic paired-t-test-vs-Wilcoxon selection via a Shapiro-Wilk
  normality check, Cohen's d, and a 95% confidence interval.
- `ReportGenerator` (`report.py`) formats a `StatisticalComparison` as
  Markdown, JSON, or a compact summary table.

No reinforcement learning, no PPO, and no learning of any kind.
`app/graph`, `app/router`, `app/policy_engine`,
`app/evaluation/offline/{replay,benchmark}.py`,
`app/evaluation/experiments`, `app/reward`, `app/experience`, and
`app/context` are all used as-is, unmodified.
"""

from app.evaluation.statistics.analyzer import (
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_NORMALITY_LEVEL,
    DEFAULT_SIGNIFICANCE_LEVEL,
    Analyzer,
)
from app.evaluation.statistics.models import StatisticalComparison
from app.evaluation.statistics.report import ReportGenerator

__all__ = [
    "DEFAULT_CONFIDENCE_LEVEL",
    "DEFAULT_NORMALITY_LEVEL",
    "DEFAULT_SIGNIFICANCE_LEVEL",
    "Analyzer",
    "ReportGenerator",
    "StatisticalComparison",
]
