> **ARCHIVED — SUPERSEDED, DO NOT CITE.** Generated before the
> target-leakage fix in the offline replay context builder. See
> `../README.md` and the current `docs/Threats_to_Validity.md` for the
> revalidated analysis.

# ACRF v1.0 Final Experimental Campaign — Threats to Validity

This campaign was executed strictly through the existing, unmodified
ACRF evaluation pipeline (`app/evaluation/offline`,
`app/evaluation/experiments`, `app/evaluation/ablation`,
`app/evaluation/statistics`, `app/evaluation/learning_analysis`). No
framework code was created, refactored, or modified — only
`experiments/campaign_v2/run_campaign.py`, a standalone driver script,
was written, and it calls nothing but existing, exported classes (plus
two small, clearly-documented aggregation-glue helper functions that
mirror `OfflineEvaluator`/`ExperimentRunner`'s own formulas using only
public building blocks, needed because neither class has a code path
for `ReplayEngine.replay_with_learning()`'s output — see the script's
"Small aggregation glue" section).

## Construct validity

- **The experience log is entirely synthetic**, identical to
  `campaign_v1`'s generator (same archetypes, same seed `12345`) for
  continuity. No live LLM or agent execution exists anywhere in this
  framework by design. Every quantitative finding should be read as "the
  evaluation pipeline correctly detects the difference this synthetic
  data was constructed to contain," not as a claim about real-world ACRF
  performance.
- **The synthetic log's cold-start-optimal archetype directly shapes
  Sequential Learning's result.** `CodeCritic` was deliberately given
  the highest quality/lowest latency archetype in the generator, and it
  also happens to be the alphabetically-first, cold-start tie-break
  winner. This is a coincidence of the generator's construction (not
  re-engineered for this campaign — it is identical to `campaign_v1`'s),
  but it directly determines Sequential Learning's finding: because the
  cold-start default was already the best available choice, any
  exploration away from it inside a single sequential pass could only
  reduce measured reward. A log where the cold-start default is *not*
  already near-optimal would likely show a different, possibly opposite,
  result — see `Discussion.md`'s "Future work."
- **Reward is itself a proxy**, with fixed weights not fitted to any
  real objective (`app/reward/strategy.py`). The Quality-only Reward
  ablation's significant finding demonstrates this choice is not
  neutral.

## Internal validity

- **Sequential Learning's per-pass irreversibility is a property of the
  evaluation mode, not a bug.** `ReplayEngine.replay_with_learning()`
  trains only on experiences where the policy's live selection matches
  the historical record — the same unbiased "replay method" rule
  `replay()` uses, and a *conservative*, correctness-preserving design
  choice (never fabricating a reward for an untaken action). One
  consequence, fully explained in `Discussion.md`, is that an arm
  abandoned mid-pass can never be revisited or corrected within that
  same pass. This campaign surfaces that consequence; it does not
  indicate an implementation defect.
- **Every non-training policy is untrained** (Heuristic, Cold-Start
  LinUCB, Random Critic, Reduced Context, Quality-only Reward candidate)
  — `ExperimentRunner`/`AblationRunner` never call `.update()`, per
  their own Research Constraints, upheld unmodified in this campaign.
- **The offline "replay method" only scores matched experiences** —
  roughly 39% of the log for most cold-start policies, and considerably
  less (8-14%) for Sequential Learning at alpha >= 0.5, once it has
  switched away from the dominant archetype. A smaller matched sample
  means a noisier `average_reward`/`match_rate` estimate for those rows
  specifically (visible as a wider spread in
  `figures/reward_distribution.png`'s "Sequential Learning" box).
- **No multiple-comparisons correction was applied.** Across Tables 1-3,
  roughly a dozen hypothesis tests were run at an uncorrected alpha =
  0.05. Every significant p-value found here is `<0.0006`, comfortably
  surviving even a conservative Bonferroni correction
  (0.05 / 12 ≈ 0.004), so this does not change this campaign's
  conclusions, but a campaign with more borderline results would need
  to correct for this.
- **The raw learning-curve statistics (Table 4, most figures) come from
  a single deterministic pass**, not the 30-run bootstrap — by design
  (mirroring `ExperimentRunner`'s own `num_runs == 1` convention), but
  it means Table 4's `convergence_step`/`learning_rate_estimate` are
  point estimates from one specific ordering of the log, not averaged
  across resamples. The bootstrap statistics in Tables 1-3 (average
  reward, CI, p-value, effect size) do not have this limitation.

## External validity

- **300 synthetic records** and **30 bootstrap resamples** are the same
  modest scale used in `campaign_v1`; a real deployment would likely
  accumulate substantially more data, which would narrow confidence
  intervals but not change the structural findings (cold-start
  invariance is a mathematical property of untrained policies, not a
  data-volume artifact; the arm-abandonment mechanism would still occur
  whenever a match-gated single pass switches arms mid-run, regardless
  of log size).
- **Only one exploration-coefficient schedule was tested**: a constant
  alpha for the entire pass. A decaying or periodically-reset schedule
  (not implemented anywhere in the frozen framework) was not evaluated
  and might behave very differently.
- **A single random seed pair** (data generator `12345`, campaign
  `2024`) was used throughout; robustness across seeds was not assessed.
  A different data-generation seed could change which archetype the
  cold-start tie-break favors and by how much, which would materially
  change Sequential Learning's specific numbers (though the qualitative
  mechanism — irreversible mid-pass arm-switching under a match-gated
  training rule — would still apply.)

## Conclusion validity

- **Every p-value, effect size, and confidence interval was computed
  exactly as `app.evaluation.statistics.Analyzer` defines them**
  (auto-selected paired t-test or Wilcoxon signed-rank via Shapiro-Wilk
  normality testing; Cohen's d for paired samples; percentile-method
  confidence intervals) — this campaign did not choose or tune the
  statistical methodology, only invoked it.
- **Determinism was verified, not merely assumed**: this script asserts
  the source repository's record count and every record's serialized
  content are unchanged before and after the full campaign, and the
  entire pipeline (bootstrap resampling, replay matching, sequential
  training, statistical testing) is seeded and reproducible end to end
  — re-running `run_campaign.py` reproduces every number in this report
  exactly.
- **Sequential Learning's regression is large and consistent across all
  four tested alpha values** (not an isolated or borderline result),
  and its magnitude scales monotonically with alpha in the direction the
  mechanistic explanation in `Discussion.md` predicts — this internal
  consistency is itself evidence the finding reflects the mechanism
  described, not noise or a computation artifact.
