"""The Learning Analysis layer: extends the existing experiment analysis
with learning-curve metrics.

Pure, **read-only** analysis of an already-completed sequential replay's
`app.evaluation.offline.ReplayStep`s (most meaningfully, the output of
`ReplayEngine.replay_with_learning`, see `app/evaluation/offline/replay.py`).

- `LearningAnalyzer` (`analyzer.py`) computes reward-per-step, cumulative
  reward, instantaneous/cumulative regret, moving-average reward,
  convergence point, and a learning-rate estimate.
- `LearningReportGenerator` (`report.py`) formats a `LearningCurve` as
  Markdown, CSV, or JSON.

No new infrastructure, no architecture changes, no changes to
`ReplayEngine`, and no changes to any policy: this module only ever
transforms an already-produced `list[ReplayStep]`. It never calls
`select_action`/`update` on anything, never reads an
`ExperienceRepository`, and never creates or stores an `ExperienceRecord`.
"""

from app.evaluation.learning_analysis.analyzer import (
    DEFAULT_CONVERGENCE_TOLERANCE,
    DEFAULT_MOVING_AVERAGE_WINDOW,
    LearningAnalyzer,
)
from app.evaluation.learning_analysis.models import LearningCurve
from app.evaluation.learning_analysis.report import LearningReportGenerator

__all__ = [
    "DEFAULT_CONVERGENCE_TOLERANCE",
    "DEFAULT_MOVING_AVERAGE_WINDOW",
    "LearningAnalyzer",
    "LearningCurve",
    "LearningReportGenerator",
]
