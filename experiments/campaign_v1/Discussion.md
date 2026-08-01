# ACRF v1.0 Experimental Campaign — Discussion

See `Results.md` for the full tables/figures this discussion refers to.

## Summary of findings

**No experiment in this campaign showed a statistically significant
*improvement* over Baseline or Full ACRF.** The two statistically
significant results found (Random Critic Selection, Quality-only
Reward) were both **regressions** relative to Full ACRF. Every other
comparison — the entire alpha sweep, and Reduced Context Features — was
statistically indistinguishable from Baseline/Full ACRF. Reporting this
honestly, rather than as a false positive "ACRF improves on the
baseline," is the correct outcome of this campaign: it is a genuine,
reproducible null result with a well-understood mechanical cause,
explained below.

## Why experiments 1–6 and 10 are identical

`HeuristicPolicy` and an **untrained** `LinUCBPolicy` (any alpha)
converge to the exact same critic on this campaign's replayed contexts,
for two independent, already-documented structural reasons:

1. **`HeuristicPolicy` scores critics from nine specifically-named
   context features** (`uncertainty`, `risk`, `task_complexity`, ...;
   see `app/policy/heuristic_policy.py`) that
   `app.evaluation.offline.build_offline_context_vector` — the context
   builder offline replay actually uses — does not produce (it cannot:
   offline replay has no live `AgentState`, only a stored
   `ExperienceRecord`; this is documented in
   `app/evaluation/offline/README.md`'s "Known limitation" section).
   Every one of those nine inputs defaults to a neutral `0.0`/`False`,
   so `HeuristicPolicy` computes the exact same score for every
   experience and always selects `CodeCritic` (ties broken
   alphabetically).
2. **An untrained `LinUCBPolicy`'s arms all start identical**
   (`A = I`, `b = 0`), so for any context every arm's upper confidence
   bound reduces to `alpha * sqrt(xᵀx)` — the *same* value for every
   arm, scaled by the *same* alpha. Scaling every arm's score by an
   identical constant never changes which arm wins a tie, which is why
   **alpha has zero effect on selection for an untrained policy** —
   `app/evaluation/experiments/README.md` documents this exact
   mechanism and predicts precisely the outcome observed here.

Because `ExperimentRunner`/`AblationRunner` are constrained to never
call `.update()` on a policy during evaluation (a Research Constraint
of the frameworks that built them, upheld in this campaign too — no
framework code was modified), every policy evaluated here is
permanently in this untrained, cold-start state. This is not a bug
introduced by this campaign; it is the direct, predictable, and (per
the frameworks' own READMEs) *expected* consequence of evaluating
untrained policies offline. The campaign's numbers simply confirm the
prediction empirically, with real bootstrap-resampled statistics rather
than a single anecdotal check.

**Practical implication:** this campaign cannot speak to whether
`LinUCBPolicy` — once actually trained — would outperform
`HeuristicPolicy`, nor to whether alpha matters for a *trained* policy.
It only establishes that, out of the box, with no training, the two
policies (and every alpha) are behaviorally and statistically identical
under this offline-replay context vocabulary. See "Future Work" below.

## Reduced Context Features: also degenerate, for the same reason

Since an untrained `LinUCBPolicy`'s selection is already
context-invariant (point 2 above — every arm's score scales identically
regardless of what `x` is), masking half the context features
(`keep_feature_fraction=0.5`) changes nothing: `ReducedContextPolicy`
still hands the (reduced) context to the same untrained arms, which
still tie identically regardless of `x`'s magnitude or dimensionality.
This ablation would only be informative against a *trained* policy,
where different arms' `A`/`b` genuinely differ and thus which features
survive reduction could plausibly change which arm wins.

## The two significant findings

**Random Critic Selection** is dramatically worse than Full ACRF
(reward −0.1273, p < 0.0001, d = −3.99 — a very large effect by any
convention). This is the one comparison in the campaign with a genuine
behavioral difference between arms (a uniform-random policy vs. an
always-`CodeCritic` policy), and it demonstrates, at minimum, that
**always routing to the single highest-quality-correlated critic in
this dataset beats routing uniformly at random** — a sensible, expected
result given how the synthetic log was generated (see
`Threats_to_Validity.md`) and a useful sanity check that the evaluation
pipeline (replay, matching, reward computation, bootstrap statistics)
is behaving correctly end-to-end.

**Quality-only Reward** differs significantly from the standard
weighted reward (reward −0.0096, p = 0.0003, d = −0.75) despite
*identical* critic selections and *identical* match rate — the
difference is attributable entirely to the reward **definition**, not
to any behavioral difference. The standard `WeightedRewardStrategy`
adds a completion bonus (`+0.2` for a `"completed"` execution) and
subtracts modest cost/latency/correction penalties; with a ~92%
completion rate and moderate latency in the synthetic log, the net
effect nudges the weighted reward above raw quality. This is a useful,
if modest, illustration that **the choice of reward definition itself
materially affects the measured value of a policy**, independent of
what the policy actually does.

## Limitations (see `Threats_to_Validity.md` for the full list)

- All data is **synthetic** — no live LLM execution exists in this
  framework by design, so the entire campaign is a demonstration of the
  evaluation pipeline's mechanics, not a live-system evaluation.
- Every policy is **untrained**, by the frameworks' own Research
  Constraints — this campaign cannot assess trained-policy behavior.
- The offline "replay method" only scores experiences where a policy's
  selection exactly matches the historical record (match rate ≈ 39% for
  most policies here); the ~61% of unmatched experiences contribute no
  signal, a known limitation of this evaluation technique in general.
- Ten hypothesis tests were run at alpha = 0.05 without a multiple-comparisons
  correction (e.g. Bonferroni); with ten tests, the family-wise false
  positive rate is inflated. Both significant findings here have p-values
  far below even a conservative corrected threshold (< 0.005), so this
  does not change the campaign's conclusions, but it should be corrected
  for in a campaign with more borderline results.

## Future work

1. **Pre-train a `LinUCBPolicy`** (via direct `.update()` calls, outside
   evaluation — the documented pattern in
   `app/evaluation/experiments/README.md`) and re-run this exact
   campaign against the trained instance. This is the single most
   informative next step: it would let alpha sensitivity, Reduced
   Context Features, and a genuine Heuristic-vs-LinUCB comparison
   produce non-degenerate results.
2. **Replace the synthetic experience log with real recorded executions**
   once a live ACRF deployment exists, removing the single largest
   threat to validity in this campaign.
3. **Extend the offline-replay context vocabulary** (a change to
   `app/evaluation/offline`, out of scope for this execution-only
   campaign and for the frozen framework) to preserve the nine
   `HeuristicPolicy`-parity signals, so `HeuristicPolicy` is no longer
   context-invariant under replay.
4. **Apply a multiple-comparisons correction** (Bonferroni or
   Benjamini-Hochberg) when running a campaign with more than a
   small handful of hypothesis tests, or with borderline p-values.
5. **Widen the alpha sweep and add exploration-regret metrics** once
   evaluating trained policies is possible, to characterize the
   explore/exploit tradeoff `alpha` is meant to control.
