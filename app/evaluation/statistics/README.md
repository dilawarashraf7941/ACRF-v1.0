# Statistical Analysis Framework

Compares two policies' experimental results scientifically: a paired
hypothesis test (chosen automatically), an effect size, a confidence
interval, and a report in three formats.

> **Scope:** statistical analysis and reporting only. No changes to
> `app/graph`, `app/router`, `app/policy_engine`,
> `app/evaluation/offline/{replay,benchmark}.py`,
> `app/evaluation/experiments` (`ExperimentRunner`), `app/reward`,
> `app/experience`, or `app/context` — all used as-is. No PPO, no
> reinforcement learning. Every input (`ExperimentResult`,
> `ReplayResult`) is a frozen Pydantic model already; nothing in this
> module calls a mutating method on anything.

## Files

| File | Responsibility |
|---|---|
| `models.py` | `StatisticalComparison` — the immutable result of one paired comparison. |
| `analyzer.py` | `Analyzer` — descriptive statistics, effect size, confidence interval, and the paired-t-test-vs-Wilcoxon decision logic. |
| `report.py` | `ReportGenerator` — Markdown, JSON, and summary-table output. |

## What "paired" means here, and why it matters

`Analyzer` compares two **equal-length, index-aligned** sequences of
values — run `0` of the baseline against run `0` of the candidate, run
`1` against run `1`, and so on. This pairing is only as statistically
meaningful as the two runs actually being comparable events.

`app.evaluation.experiments.ExperimentRunner` draws its bootstrap
resamples from a single `random.Random(config.random_seed)`, consumed
sequentially across all `num_runs` runs. Two `ExperimentConfig`s that
share the same `random_seed` and `num_runs` therefore see run `i`
replay the **identical** resample of the source data under each policy
— a genuine matched pair (same data, different policy), exactly what a
paired test assumes. `Analyzer.compare_experiments` checks this and
records it under `metadata["same_random_seed"]`; it does not *require*
a seed match (index-aligned pairing is still computed either way), but
a comparison built from mismatched seeds is comparing two independent
resamples index-by-index, which is a materially weaker claim than a
true matched pair — read `same_random_seed` before trusting the result.

## Decision logic

For each paired comparison, in order:

1. **`sample_size == 1`** — a single observation cannot support any
   hypothesis test. `test_used = "insufficient_data"`, `p_value = 1.0`.
2. **Zero variance** — every paired difference is exactly identical
   (including the common case of two identical samples, where every
   difference is `0`). Neither test's assumptions are meaningful when
   there is no variability at all; `test_used = "degenerate_zero_variance"`,
   `p_value = 1.0` if the differences are all `0`, `0.0` otherwise (the
   difference is then certain, not merely likely — the limiting case of
   either test as variance shrinks to zero).
3. **Otherwise**, a Shapiro-Wilk normality test runs on the paired
   differences:
   - `p > normality_level` (default `0.05`, i.e. normality is not
     rejected) → **paired t-test** (`scipy.stats.ttest_rel`).
   - Otherwise → **Wilcoxon signed-rank test**
     (`scipy.stats.wilcoxon`), the nonparametric alternative.
   - Shapiro-Wilk itself needs at least 3 observations; below that, the
     test can't run at all, and the framework conservatively defaults
     to Wilcoxon (`is_normal = False`) rather than assuming normality
     it cannot check.

Every branch is recorded on `StatisticalComparison.test_used` and
explained further in `metadata` (`"reason"` for the two degenerate
branches, `"normality_test"`/`"normality_p_value"`/
`"normality_test_skipped_reason"` for the third).

## Effect size and confidence interval

- **Effect size** is Cohen's d for paired samples (`d_z = mean(differences)
  / std_dev(differences)`) — standardized by the spread of the
  difference scores themselves, the standard choice for a matched-pairs
  design (as opposed to a pooled two-independent-groups SD). `0.0` when
  there are fewer than two paired observations or the differences have
  zero variance.
- **Confidence interval** is always the **t-distribution-based** interval
  for the mean difference (`mean_difference ± t_(n-1, 0.975) *
  std_dev/sqrt(n)`), regardless of whether `test_used` ends up being
  `paired_t_test` or `wilcoxon_signed_rank`. This is a deliberate
  simplification: a single, standard, deterministic CI for the point
  estimate, rather than introducing a second, less-standard
  nonparametric interval (e.g. Hodges-Lehmann) alongside Wilcoxon that
  nothing in this task asked for.

## `Analyzer`

```python
analyzer = Analyzer()  # significance_level=0.05, normality_level=0.05, confidence_level=0.95

# Generic: any two equal-length numeric sequences.
comparison = analyzer.compare_samples(baseline_values, candidate_values,
                                       baseline_policy="HeuristicPolicy",
                                       candidate_policy="LinUCBPolicy")

# Convenience: reads a named ReplayResult field off each run, reflectively —
# works for any current or future numeric field, and any policy, without
# modification (no policy class is ever imported by this module).
comparison = analyzer.compare_experiments(baseline_result, candidate_result,
                                           metric="average_reward")
```

`compare_samples` is the general-purpose entry point (works on any
paired float sequences — e.g. from a source outside this framework
entirely); `compare_experiments` is a thin wrapper that extracts a
named `ReplayResult` field from each `ExperimentResult.runs` via
`getattr` and delegates to it. Individual statistics (`mean`,
`standard_deviation`, `mean_difference`, `cohens_d`,
`confidence_interval`, `is_normally_distributed`, `paired_t_test`,
`wilcoxon_signed_rank`) are all public methods too, independently
usable and independently testable.

> **Known interaction with earlier limitations.** As documented in
> `app/evaluation/experiments/README.md`, an untrained `LinUCBPolicy`
> and `HeuristicPolicy` both deterministically select the same critic
> under the offline-replay `ContextVector` vocabulary. Comparing two
> such untrained policies over the same seed will typically produce
> `test_used = "degenerate_zero_variance"` with `mean_difference = 0.0`
> — every paired run matched identically, so there is nothing to test.
> This is not a flaw in the statistics; it faithfully reports that
> nothing differed. A meaningful, non-degenerate comparison requires an
> actually-differentiated candidate policy (e.g. a pre-trained
> `LinUCBPolicy`, per that README's documented pattern).

## `ReportGenerator`

```python
report = ReportGenerator()
report.to_markdown(comparison)                  # full narrative report
report.to_json(comparison)                       # machine-readable
report.to_summary_table([comparison, ...])       # compact multi-comparison table
```

`to_markdown` includes a hypothesis statement (H0/H1), which test was
used, the p-value, the effect size (with its Cohen's-d
negligible/small/medium/large label), the confidence interval, a
plain-language interpretation of which policy scored higher, and a
conclusion sentence stating significance against the configured
threshold. `to_summary_table` renders one row per comparison — useful
for reporting a sweep (e.g. one row per `alpha` value) at a glance.

## Explicit non-goals

- No changes to `app/graph`, `app/router`, `app/policy_engine`,
  `ReplayEngine`, `Benchmark`, `ExperimentRunner`, `app/reward`,
  `app/experience`, or `app/context`.
- No mutation of any `ExperimentResult`/`ReplayResult` — every read is
  via plain attribute access on an already-frozen model.
- No PPO, no reinforcement learning.
- No randomness: `Analyzer` never seeds or calls anything random;
  identical inputs always produce a bit-identical `StatisticalComparison`.
- No hardcoded policy names anywhere in `analyzer.py`/`report.py` — a
  future policy's `ExperimentResult` works with this framework
  unmodified, as long as its `ReplayResult`s expose the numeric field
  being compared.
