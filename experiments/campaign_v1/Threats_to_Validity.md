# ACRF v1.0 Experimental Campaign — Threats to Validity

This campaign was executed strictly through the existing, unmodified
ACRF evaluation pipeline (`app/evaluation/offline`, `app/evaluation/experiments`,
`app/evaluation/ablation`, `app/evaluation/statistics`). No framework
code was created, refactored, or modified — only `experiments/campaign_v1/run_campaign.py`,
a standalone driver script, was written, and it calls nothing but
existing, exported classes. That scope constraint is itself the source
of several of the threats below: this campaign inherits every
already-documented limitation of the frameworks it calls, and cannot
work around any of them without violating "do not modify the
framework."

## Construct validity

- **The experience log is entirely synthetic.** No live LLM or agent
  execution exists anywhere in this framework by design (see
  `app/evaluation/offline/README.md`: "No LLM execution. No graph
  execution."). `run_campaign.py::generate_repository` produces 300
  `ExperienceRecord`s from five hand-tuned "task archetype" distributions
  (fixed weights, Gaussian quality/latency, seed `12345`) that
  *deliberately* correlate `CodeCritic` selection with higher quality
  and lower latency/iterations, specifically so the campaign would have
  *some* non-degenerate signal to report (see `random_critic_selection`'s
  large effect size in `Results.md`). **This is a designed proxy for
  realistic data, not a measurement of any real system.** Every
  quantitative finding in `Results.md`/`Discussion.md` should be read as
  "the evaluation pipeline correctly detects the difference this
  synthetic data was constructed to contain," not as a claim about
  real-world ACRF performance.
- **Reward is itself a proxy.** `WeightedRewardStrategy`'s weights
  (quality weight, cost/latency/correction penalty scales, completion
  bonus) are fixed constants chosen at implementation time (see
  `app/reward/strategy.py`), not fitted to any real objective. The
  "Quality-only Reward" ablation's significant finding
  (`Results.md`, Table 2) demonstrates that this choice of weights is
  *not* neutral — different reasonable reward definitions would report
  different absolute numbers for the identical underlying behavior.

## Internal validity

- **Every evaluated policy is untrained.** `ExperimentRunner`/`AblationRunner`
  never call `.update()` on any policy (a Research Constraint of both
  frameworks). As `Discussion.md` explains in detail, this is the
  direct, mechanical cause of the campaign's central finding (Baseline
  = all 5 alphas = Full ACRF, bit-for-bit identical). **This campaign
  cannot and does not claim anything about trained-policy behavior.**
- **The offline "replay method" only scores matched experiences.**
  Per `app/evaluation/offline/README.md`, an experience only contributes
  to a policy's statistics if that policy's selection exactly matches
  the historically recorded critic(s). Match rate was ≈ 39% for every
  policy except Random Critic (≈ 22%) — meaning roughly 60–78% of the
  300-record log contributed *no* signal to any given policy's reported
  numbers. This is a standard, known limitation of offline off-policy
  evaluation by the replay method (Li et al., 2011), not specific to
  this campaign, but it does mean effective sample sizes per bootstrap
  run were smaller than the raw 300-record log might suggest.
- **`Reduced Context Features` was only tested against `LinUCBPolicy`**,
  not `HeuristicPolicy` (which has no context-feature dependency to
  reduce under this offline vocabulary in the first place — see
  `Discussion.md`). The ablation's degenerate result is fully explained
  by the untrained-policy issue above, not by the feature-reduction
  mechanism itself being ineffective in general.
- **No multiple-comparisons correction was applied.** Ten hypothesis
  tests were run at an uncorrected alpha = 0.05 (Table 1). Both
  significant p-values found (< 0.0001, 0.0003) comfortably survive
  even a conservative Bonferroni correction (0.05 / 10 = 0.005), so
  this does not change this campaign's conclusions, but a campaign with
  more borderline results would need to correct for this.

## External validity

- **300 synthetic records is a modest log size** for a production
  routing system; real ACRF deployments would likely accumulate orders
  of magnitude more experience data, which would change bootstrap
  confidence interval widths (narrower, generally) but not the
  structural cold-start finding.
- **`num_runs=30` bootstrap resamples** is a reasonable but arbitrary
  choice for confidence interval resolution; it was not tuned or
  justified via a formal power analysis.
- **The candidate action set is fixed at the framework's four built-in
  critics** (`LogicCritic`, `CodeCritic`, `FactCritic`, `MetaCritic`);
  findings do not generalize to a system with a different critic roster.
- **A single random seed pair** (data generator `12345`, campaign
  `2024`) was used throughout. Reproducibility is exact given these
  seeds (verified: the source repository's record count was asserted
  unchanged before and after every run), but robustness *across* seeds
  was not assessed — a different data-generation seed could produce a
  different quality/critic correlation strength and thus a different
  Random Critic effect size, though the *structural* degeneracy finding
  (points 1–6, 10 identical) would hold regardless of seed, since it is
  a mathematical property of untrained policies, not a data artifact.

## Conclusion validity

- **Effect sizes and p-values were computed exactly as the
  `app.evaluation.statistics.Analyzer` module defines them** (paired
  t-test or Wilcoxon signed-rank, auto-selected via Shapiro-Wilk
  normality testing on paired differences; Cohen's d for paired
  samples) — this campaign did not choose or tune the statistical
  methodology, only invoked it, so any critique of the *methodology
  itself* (e.g. the choice of percentile-method confidence intervals
  over Hodges-Lehmann, documented in
  `app/evaluation/statistics/README.md`) applies equally here and was
  not re-litigated.
- **Six of ten comparisons resolved to the framework's
  `degenerate_zero_variance` test** (every one of 30 paired differences
  being *exactly* zero), which is a legitimate, correctly-reported
  outcome for genuinely identical paired samples — not a statistical
  artifact or a computation error. This was verified independently: the
  campaign script asserts (and this held) that the independently
  replayed candidate arm for each ablation matches
  `AblationRunner`'s own internal candidate arm's `average_reward`
  exactly, confirming the pipeline's own determinism guarantee held
  throughout.
