"""`Analyzer`: paired statistical comparison between two policies' results.

Compares two equal-length sequences of paired observations — most
naturally, two `ExperimentResult`s' per-run metric values (see
`compare_experiments`), e.g. `HeuristicPolicy`'s and `LinUCBPolicy`'s
`average_reward` across runs `0..N-1`. The pairing is only statistically
meaningful when run `i` of both experiments reflects the same underlying
data: `app.evaluation.experiments.ExperimentRunner` draws its bootstrap
resamples from a `random.Random(config.random_seed)` consumed
sequentially, so two `ExperimentConfig`s sharing the same `random_seed`
and `num_runs` see run `i` replay the *identical* resample under each
policy — `compare_experiments` records whether that held via
`metadata["same_random_seed"]`, but does not require it (index-aligned
pairing is still computed either way; only the methodological strength
differs).

No reinforcement learning, no PPO, and no learning of any kind. This
module reads `ExperimentResult`/`ReplayResult` (already produced
elsewhere) and never mutates them — every input model is frozen, and
nothing here ever calls `.add()`, `.update()`, or any other mutating
method on anything.
"""

import math
from collections.abc import Sequence
from typing import Any

from scipy import stats

from app.evaluation.experiments import ConfidenceInterval, ExperimentResult
from app.evaluation.statistics.models import StatisticalComparison

DEFAULT_SIGNIFICANCE_LEVEL = 0.05
"""The p-value threshold `significant` is computed against, by default."""

DEFAULT_NORMALITY_LEVEL = 0.05
"""The Shapiro-Wilk p-value threshold above which differences are treated as normal."""

DEFAULT_CONFIDENCE_LEVEL = 0.95
"""The confidence level `confidence_interval` uses by default."""

MIN_SAMPLES_FOR_NORMALITY_TEST = 3
"""Shapiro-Wilk requires at least this many observations to be computable at all."""


class Analyzer:
    """Runs a paired statistical comparison and selects the appropriate test.

    Decision logic: if the paired differences pass a Shapiro-Wilk
    normality test (`p > normality_level`), use a paired t-test;
    otherwise use the Wilcoxon signed-rank test. Two further cases are
    handled explicitly, before that decision is ever reached: a single
    paired observation (`sample_size == 1`, no test is meaningful) and
    zero-variance differences (every pair differs by exactly the same
    amount — the mathematically degenerate limit of either test).

    Every method is a pure function of its arguments plus this
    instance's `significance_level`/`normality_level`/`confidence_level`
    — no state is mutated, and no randomness is used anywhere, so
    calling any method twice on the same inputs always returns an
    identical result.
    """

    def __init__(
        self,
        significance_level: float = DEFAULT_SIGNIFICANCE_LEVEL,
        normality_level: float = DEFAULT_NORMALITY_LEVEL,
        confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    ) -> None:
        """Create an analyzer with the given thresholds.

        Args:
            significance_level: The p-value threshold `significant` is
                computed against (`StatisticalComparison.significant =
                p_value < significance_level`).
            normality_level: The Shapiro-Wilk p-value threshold above
                which paired differences are treated as normally
                distributed (selecting the paired t-test).
            confidence_level: The confidence level `confidence_interval`
                computes by default.

        Raises:
            ValueError: If any threshold is not strictly between 0 and 1.
        """
        for name, value in (
            ("significance_level", significance_level),
            ("normality_level", normality_level),
            ("confidence_level", confidence_level),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be strictly between 0 and 1, got {value}")

        self._significance_level = significance_level
        self._normality_level = normality_level
        self._confidence_level = confidence_level

    # --- basic descriptive statistics -------------------------------------------------

    @staticmethod
    def mean(values: Sequence[float]) -> float:
        """Return the arithmetic mean of `values`, or `0.0` for an empty sequence."""
        values = list(values)
        return sum(values) / len(values) if values else 0.0

    def standard_deviation(self, values: Sequence[float]) -> float:
        """Return the sample standard deviation (`ddof=1`) of `values`.

        Returns `0.0` for fewer than two values, matching
        `app.evaluation.experiments.Analyzer.std_dev`'s convention.
        """
        values = list(values)
        if len(values) < 2:
            return 0.0
        mean = self.mean(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        return math.sqrt(variance)

    def mean_difference(
        self, baseline_values: Sequence[float], candidate_values: Sequence[float]
    ) -> float:
        """Return `mean(candidate_values) - mean(baseline_values)`."""
        return self.mean(candidate_values) - self.mean(baseline_values)

    @staticmethod
    def _paired_differences(
        baseline_values: Sequence[float], candidate_values: Sequence[float]
    ) -> list[float]:
        return [c - b for b, c in zip(baseline_values, candidate_values, strict=True)]

    # --- effect size and confidence interval -------------------------------------------

    def cohens_d(
        self, baseline_values: Sequence[float], candidate_values: Sequence[float]
    ) -> float:
        """Compute Cohen's d for paired samples (`d_z`).

        `d_z = mean(differences) / std_dev(differences)` — the standard
        paired-samples variant, standardizing by the spread of the
        difference scores themselves rather than a pooled two-group SD.

        Args:
            baseline_values: The baseline policy's per-run values.
            candidate_values: The candidate policy's per-run values,
                paired index-for-index with `baseline_values`.

        Returns:
            `d_z`, or `0.0` when fewer than two paired observations
            exist or the differences have zero variance (there is no
            meaningful spread to standardize against in either case).
        """
        differences = self._paired_differences(baseline_values, candidate_values)
        if len(differences) < 2:
            return 0.0
        std_dev = self.standard_deviation(differences)
        if std_dev == 0.0:
            return 0.0
        return self.mean(differences) / std_dev

    def confidence_interval(
        self,
        baseline_values: Sequence[float],
        candidate_values: Sequence[float],
        confidence_level: float | None = None,
    ) -> ConfidenceInterval:
        """Compute a t-distribution-based confidence interval for the mean difference.

        Used for `mean_difference` regardless of whether `test_used`
        ends up being `'paired_t_test'` or `'wilcoxon_signed_rank'` — a
        single, standard, deterministic CI for the point estimate,
        rather than a second, less-standard nonparametric CI method
        (e.g. Hodges-Lehmann) alongside Wilcoxon.

        Args:
            baseline_values: The baseline policy's per-run values.
            candidate_values: The candidate policy's per-run values,
                paired index-for-index with `baseline_values`.
            confidence_level: Overrides this analyzer's configured
                `confidence_level` for this call only.

        Returns:
            A `ConfidenceInterval`. Degenerates to a point interval
            (`lower == upper == mean_difference`) for fewer than two
            paired observations, or zero-variance differences.
        """
        level = confidence_level if confidence_level is not None else self._confidence_level
        differences = self._paired_differences(baseline_values, candidate_values)
        mean_diff = self.mean(differences)

        if len(differences) < 2:
            return ConfidenceInterval(lower=mean_diff, upper=mean_diff, confidence_level=level)

        std_dev = self.standard_deviation(differences)
        if std_dev == 0.0:
            return ConfidenceInterval(lower=mean_diff, upper=mean_diff, confidence_level=level)

        n = len(differences)
        standard_error = std_dev / math.sqrt(n)
        t_critical = float(stats.t.ppf(1.0 - (1.0 - level) / 2.0, df=n - 1))
        margin = t_critical * standard_error
        return ConfidenceInterval(
            lower=mean_diff - margin, upper=mean_diff + margin, confidence_level=level
        )

    # --- normality and hypothesis tests -------------------------------------------------

    def is_normally_distributed(self, values: Sequence[float]) -> tuple[bool, dict[str, Any]]:
        """Test whether `values` are plausibly normally distributed (Shapiro-Wilk).

        Args:
            values: The values to test — normally the paired differences.

        Returns:
            A `(is_normal, metadata)` pair. `is_normal` is `True` when
            the Shapiro-Wilk p-value exceeds this analyzer's
            `normality_level`. For fewer than `MIN_SAMPLES_FOR_NORMALITY_TEST`
            values, Shapiro-Wilk cannot be computed at all; `is_normal`
            is conservatively `False` (defaulting to the nonparametric
            Wilcoxon test) and `metadata` explains why.
        """
        values = list(values)
        if len(values) < MIN_SAMPLES_FOR_NORMALITY_TEST:
            return False, {
                "normality_test": "shapiro_wilk",
                "normality_test_skipped_reason": (
                    f"sample_size {len(values)} < {MIN_SAMPLES_FOR_NORMALITY_TEST}; "
                    "defaulting to the nonparametric Wilcoxon signed-rank test"
                ),
            }

        statistic, p_value = stats.shapiro(values)
        return float(p_value) > self._normality_level, {
            "normality_test": "shapiro_wilk",
            "normality_statistic": float(statistic),
            "normality_p_value": float(p_value),
        }

    @staticmethod
    def paired_t_test(
        baseline_values: Sequence[float], candidate_values: Sequence[float]
    ) -> float:
        """Return the paired t-test p-value for `candidate_values` vs `baseline_values`."""
        _statistic, p_value = stats.ttest_rel(list(candidate_values), list(baseline_values))
        return float(p_value)

    @staticmethod
    def wilcoxon_signed_rank(
        baseline_values: Sequence[float], candidate_values: Sequence[float]
    ) -> float:
        """Return the Wilcoxon signed-rank test p-value for `candidate_values` vs `baseline_values`.

        Returns `1.0` (no evidence of a difference) instead of raising
        if scipy cannot compute the test for this input (e.g. every
        difference is zero after its default zero-handling).
        """
        try:
            _statistic, p_value = stats.wilcoxon(list(candidate_values), list(baseline_values))
        except ValueError:
            return 1.0
        return float(p_value)

    # --- orchestration -------------------------------------------------------------------

    def compare_samples(
        self,
        baseline_values: Sequence[float],
        candidate_values: Sequence[float],
        *,
        baseline_policy: str = "baseline",
        candidate_policy: str = "candidate",
    ) -> StatisticalComparison:
        """Run the full paired comparison and decision logic.

        Args:
            baseline_values: The baseline policy's per-run values.
            candidate_values: The candidate policy's per-run values,
                paired index-for-index with `baseline_values`.
            baseline_policy: Label recorded on the result.
            candidate_policy: Label recorded on the result.

        Returns:
            The resulting `StatisticalComparison`.

        Raises:
            ValueError: If the two sequences have different lengths, or
                both are empty.
        """
        baseline_values = list(baseline_values)
        candidate_values = list(candidate_values)
        if len(baseline_values) != len(candidate_values):
            raise ValueError(
                "baseline_values and candidate_values must be the same length for a "
                f"paired comparison (got {len(baseline_values)} and {len(candidate_values)})."
            )
        sample_size = len(baseline_values)
        if sample_size == 0:
            raise ValueError("cannot compare empty samples.")

        differences = self._paired_differences(baseline_values, candidate_values)
        mean_diff = self.mean(differences)
        std_dev = self.standard_deviation(differences)
        confidence_interval = self.confidence_interval(baseline_values, candidate_values)
        effect_size = self.cohens_d(baseline_values, candidate_values)

        metadata: dict[str, Any] = {
            "significance_level": self._significance_level,
            "std_dev": std_dev,
        }

        if sample_size == 1:
            test_used = "insufficient_data"
            p_value = 1.0
            metadata["reason"] = "a single paired observation cannot support a hypothesis test"
        elif std_dev == 0.0:
            test_used = "degenerate_zero_variance"
            p_value = 1.0 if mean_diff == 0.0 else 0.0
            metadata["reason"] = "every paired difference is identical (zero variance)"
        else:
            is_normal, normality_metadata = self.is_normally_distributed(differences)
            metadata.update(normality_metadata)
            if is_normal:
                test_used = "paired_t_test"
                p_value = self.paired_t_test(baseline_values, candidate_values)
            else:
                test_used = "wilcoxon_signed_rank"
                p_value = self.wilcoxon_signed_rank(baseline_values, candidate_values)

        return StatisticalComparison(
            baseline_policy=baseline_policy,
            candidate_policy=candidate_policy,
            sample_size=sample_size,
            mean_difference=mean_diff,
            confidence_interval=confidence_interval,
            p_value=p_value,
            effect_size=effect_size,
            test_used=test_used,
            significant=p_value < self._significance_level,
            metadata=metadata,
        )

    def compare_experiments(
        self,
        baseline: ExperimentResult,
        candidate: ExperimentResult,
        metric: str = "average_reward",
    ) -> StatisticalComparison:
        """Compare two `ExperimentResult`s' per-run values for `metric`, paired by run index.

        Reads `metric` off each of `baseline.runs`/`candidate.runs`
        (`ReplayResult` instances) reflectively via `getattr` — any
        current or future numeric `ReplayResult` field works
        unmodified, and so does any current or future policy, since
        this method never imports or references a concrete policy
        class.

        Args:
            baseline: The reference policy's `ExperimentResult`.
            candidate: The policy being evaluated against `baseline`.
            metric: The `ReplayResult` field to compare, e.g.
                `'average_reward'`, `'average_quality'`,
                `'average_latency'`, `'average_iterations'`.

        Returns:
            The resulting `StatisticalComparison`, with
            `metadata["metric"]` and `metadata["same_random_seed"]`
            (whether both experiments' `metadata["random_seed"]`
            matched — the condition under which run `i` of each
            reflects the identical bootstrap resample, making the
            pairing methodologically strongest) recorded in addition to
            `compare_samples`'s usual metadata.

        Raises:
            ValueError: If `baseline.runs` and `candidate.runs` differ
                in length, both are empty, or `metric` is not a numeric
                field on `ReplayResult`.
        """
        baseline_values = self._extract_metric(baseline, metric)
        candidate_values = self._extract_metric(candidate, metric)
        comparison = self.compare_samples(
            baseline_values,
            candidate_values,
            baseline_policy=baseline.policy_name,
            candidate_policy=candidate.policy_name,
        )

        same_seed = (
            baseline.metadata.get("random_seed") is not None
            and baseline.metadata.get("random_seed") == candidate.metadata.get("random_seed")
        )
        extra_metadata = {
            "metric": metric,
            "baseline_experiment_name": baseline.experiment_name,
            "candidate_experiment_name": candidate.experiment_name,
            "same_random_seed": same_seed,
        }
        return comparison.model_copy(
            update={"metadata": {**comparison.metadata, **extra_metadata}}
        )

    @staticmethod
    def _extract_metric(result: ExperimentResult, metric: str) -> list[float]:
        values: list[float] = []
        for run in result.runs:
            if not hasattr(run, metric):
                raise ValueError(f"ReplayResult has no field {metric!r}.")
            value = getattr(run, metric)
            if not isinstance(value, int | float):
                raise ValueError(
                    f"ReplayResult.{metric} is not numeric (got {type(value).__name__})."
                )
            values.append(float(value))
        return values
