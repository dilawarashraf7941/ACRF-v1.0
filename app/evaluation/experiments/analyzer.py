"""`Analyzer`: statistics over a set of per-run values.

Pure computation only — no experiment execution (that is
`ExperimentRunner`'s job, see `runner.py`) and no export logic (that is
`Exporter`'s job, see `exporter.py`). No reinforcement learning, no PPO,
and no learning of any kind.
"""

from collections.abc import Sequence

import numpy as np

from app.evaluation.experiments.models import ConfidenceInterval, StatisticalSummary
from app.evaluation.offline import ReplayResult

DEFAULT_CONFIDENCE_LEVEL = 0.95
"""The confidence level `Analyzer.confidence_interval`/`summarize` use when none is given."""


class Analyzer:
    """Computes mean, standard deviation, min, max, confidence intervals, and trends.

    Every method is a pure function of its arguments — no state, no
    randomness, no dependency on `ExperimentRunner`, `ReplayEngine`, or
    any policy. `confidence_interval` uses the empirical **percentile
    method** (the 2.5th/97.5th percentile of the values themselves,
    computed via `numpy.percentile`) rather than a normal-distribution
    approximation: it makes no assumption about how the values are
    distributed, which is the standard, correct pairing for values
    produced by bootstrap resampling (see `runner.py`'s module
    docstring) — there is no reason to assume a per-run average reward
    is normally distributed.
    """

    def mean(self, values: Sequence[float]) -> float:
        """Return the arithmetic mean of `values`, or `0.0` for an empty sequence."""
        if not values:
            return 0.0
        return float(np.mean(values))

    def std_dev(self, values: Sequence[float]) -> float:
        """Return the sample standard deviation (`ddof=1`) of `values`.

        Returns `0.0` for fewer than two values — a single observation
        (or none) has no meaningful spread, and `ddof=1` would otherwise
        divide by zero.
        """
        if len(values) < 2:
            return 0.0
        return float(np.std(values, ddof=1))

    def minimum(self, values: Sequence[float]) -> float:
        """Return the minimum of `values`, or `0.0` for an empty sequence."""
        if not values:
            return 0.0
        return float(min(values))

    def maximum(self, values: Sequence[float]) -> float:
        """Return the maximum of `values`, or `0.0` for an empty sequence."""
        if not values:
            return 0.0
        return float(max(values))

    def confidence_interval(
        self, values: Sequence[float], confidence_level: float = DEFAULT_CONFIDENCE_LEVEL
    ) -> ConfidenceInterval:
        """Compute an empirical percentile confidence interval over `values`.

        Args:
            values: The per-run (or otherwise repeated) observations.
            confidence_level: The interval's confidence level, e.g. `0.95`
                for a 95% interval (percentiles 2.5 and 97.5).

        Returns:
            A `ConfidenceInterval`. For zero values, `lower=upper=0.0`.
            For exactly one value, the interval degenerates to
            `lower=upper=that value` — there is no spread to estimate
            from a single observation.
        """
        if not values:
            return ConfidenceInterval(lower=0.0, upper=0.0, confidence_level=confidence_level)
        if len(values) == 1:
            point = float(values[0])
            return ConfidenceInterval(lower=point, upper=point, confidence_level=confidence_level)

        tail = (1.0 - confidence_level) / 2.0
        lower, upper = np.percentile(values, [100.0 * tail, 100.0 * (1.0 - tail)])
        return ConfidenceInterval(
            lower=float(lower), upper=float(upper), confidence_level=confidence_level
        )

    def summarize(
        self, values: Sequence[float], confidence_level: float = DEFAULT_CONFIDENCE_LEVEL
    ) -> StatisticalSummary:
        """Compute mean, standard deviation, min, max, and a confidence interval in one call.

        Args:
            values: The per-run (or otherwise repeated) observations.
            confidence_level: Passed through to `confidence_interval`.

        Returns:
            The resulting `StatisticalSummary`.
        """
        values = list(values)
        return StatisticalSummary(
            mean=self.mean(values),
            std_dev=self.std_dev(values),
            minimum=self.minimum(values),
            maximum=self.maximum(values),
            confidence_interval=self.confidence_interval(values, confidence_level),
            sample_size=len(values),
        )

    def reward_trend(self, runs: Sequence[ReplayResult]) -> list[float]:
        """Return each run's `average_reward`, in run order."""
        return [run.average_reward for run in runs]

    def quality_trend(self, runs: Sequence[ReplayResult]) -> list[float]:
        """Return each run's `average_quality`, in run order."""
        return [run.average_quality for run in runs]

    def latency_trend(self, runs: Sequence[ReplayResult]) -> list[float]:
        """Return each run's `average_latency`, in run order."""
        return [run.average_latency for run in runs]
