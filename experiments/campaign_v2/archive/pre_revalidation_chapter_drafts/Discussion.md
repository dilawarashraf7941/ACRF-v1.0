> **ARCHIVED — SUPERSEDED, DO NOT CITE.** Generated before the
> target-leakage fix in the offline replay context builder; every
> claim below is invalid. See `../README.md` and the current
> `docs/Discussion.md` for the revalidated discussion.

# ACRF v1.0 Final Experimental Campaign — Discussion

See `Results.md` for the full tables/figures this discussion refers to.

## Summary of findings

**Sequential Learning LinUCB does not improve performance in this
campaign — at every tested alpha, it performs significantly *worse*
than Cold-Start LinUCB, and the regression grows with alpha.** This is
reported plainly, not softened: it would be inaccurate and misleading
to describe Sequential Learning as an improvement here. The other
repeated finding — that Heuristic Policy, all Cold-Start LinUCB alpha
values, and Reduced Context are statistically indistinguishable — is
unchanged from prior campaigns and is explained in full there; this
discussion focuses on the new result, Sequential Learning, since that
is what this campaign was specifically designed to evaluate.

## Why Sequential Learning does not improve performance here

The cause is directly visible in Table 3/Table 4 and is mechanistic,
not incidental:

1. **An untrained `LinUCBPolicy`'s arms start identical**, so ties break
   alphabetically and every cold-start policy in this campaign
   deterministically selects `CodeCritic` — the archetype this
   synthetic log gives the highest quality and lowest latency/iterations
   (see `Threats_to_Validity.md`). Cold-Start LinUCB therefore matches
   `CodeCritic`'s ~40% share of the log on every one of its 120 matched
   steps, at a consistently high reward.
2. **`ReplayEngine.replay_with_learning()` trains only on matched
   experiences**, updating `policy.update(context, "CodeCritic", reward)`
   for every `CodeCritic` match — exactly the same unbiased "replay
   method" matching rule `replay()` uses, so no fabricated
   counterfactual reward is ever used for training. But `CodeCritic`'s
   reward in this log is noisy (`quality ~ N(0.78, 0.08)`, further
   perturbed per critic score), so a below-average `CodeCritic` reward
   early in the resampled sequence pulls `CodeCritic`'s
   `expected_reward` estimate down. The higher `alpha` is, the more the
   *other*, still-untouched arms' constant exploration bonus
   (`alpha * sqrt(xᵀx)`, identical for every untrained arm) outweighs
   `CodeCritic`'s now-lower estimate — so a **higher alpha causes the
   policy to abandon `CodeCritic` earlier and more readily**. This is
   exactly what Table 3/4 show: convergence step (effectively, "when the
   policy locks onto its new stable choice") drops from step 119 at
   alpha=0.25 to step 23 at alpha=1.0-2.0.
3. **Once switched, the abandoned arm is never revisited.** The policy's
   own selection now differs from `CodeCritic`, so the match rule in
   step 2 above stops matching `CodeCritic` experiences at all —
   `CodeCritic`'s arm receives no further updates, for better or worse,
   for the rest of the run. Whichever arm the policy switched to
   (`FactCritic` or another lower-quality archetype in this log) is now
   the *only* arm receiving updates, and there is no mechanism in this
   evaluation mode to switch back even if that arm turns out worse on
   average — which, per this log's construction, every non-`CodeCritic`
   single-critic archetype is.
4. **The match rate collapse compounds the effect.** `CodeCritic` alone
   accounts for ~40% of the log; the arm the policy switches to accounts
   for a smaller share (`LogicCritic` 20%, `FactCritic` 15%,
   `MetaCritic` 10%). Once switched, Sequential Learning's matched-step
   count drops from ~120 (matching Cold-Start) to 24-41 — a smaller,
   lower-quality-on-average sample drives both a lower `average_reward`
   and a lower `match_rate` (Table 1: 0.0823 vs. 0.3931 at alpha=1.0).

None of this is a flaw in `ReplayEngine.replay_with_learning()`,
`LinUCBPolicy`, or the reward calculation — every one of those
components is behaving exactly as designed and as documented (the
"replay method" is deliberately conservative about not fabricating
counterfactual rewards, and `LinUCBPolicy`'s exploration bonus is
deliberately designed to favor under-explored arms). The result is a
genuine, correctly-computed consequence of combining (a) a training
procedure that can never revisit an abandoned arm within one pass, with
(b) a synthetic log where the cold-start default (`CodeCritic`) happens
to already be the best-performing choice, so *any* exploration away
from it is a net loss inside a single sequential pass.

## Where might Sequential Learning help instead?

This campaign's design cannot answer that question directly, but the
mechanism above suggests concrete conditions under which it likely
would:

- **A log where the cold-start default is *not* already close to
  optimal** — sequential learning's entire value proposition is
  discovering a better arm than the naive tie-break; this campaign's
  synthetic log was constructed (see `Threats_to_Validity.md`) such that
  the tie-break winner (`CodeCritic`) already is the best archetype, so
  there was nothing better to discover.
- **Multiple passes, or a training procedure that revisits abandoned
  arms** (e.g. periodic re-exploration, or a smaller alpha that decays
  over time rather than staying fixed) — the current mode is a single,
  irreversible pass with a constant alpha; nothing in this campaign
  tested a decaying or periodically-reset exploration schedule.
- **A larger log**, so that even a smaller matched-step count for the
  post-switch arm remains a statistically adequate sample — at 24-41
  matched steps (alpha 1.0-2.0), the sequential arm's own
  `average_reward` estimate is itself fairly noisy (see the wide
  interquartile range for "Sequential Learning" in
  `figures/reward_distribution.png`).

## Limitations

See `Threats_to_Validity.md` for the complete list. In brief: all data
is synthetic; the "replay method"'s match-then-train rule, while
unbiased, is also why the arm-abandonment mechanism above has no
recovery path within a single pass; ten-plus hypothesis tests were run
without a multiple-comparisons correction (every significant p-value
here is nonetheless far below even a conservative corrected threshold).

## Future work

1. **Test a decaying or periodically-reset alpha schedule** for
   Sequential Learning, so exploration doesn't stay permanently high
   after the policy has already found a good arm — directly targeting
   the mechanism identified above. (Would require extending
   `LinUCBPolicy`/`ReplayEngine`, out of scope for this frozen,
   execution-only campaign.)
2. **Run Sequential Learning over multiple sequential passes** (repeated
   epochs over the same log) rather than one pass, to test whether a
   poor early switch gets corrected given more data — again out of
   scope for the current, single-pass `replay_with_learning()`.
3. **Construct a synthetic log (or use a real one) where the cold-start
   default is deliberately *not* optimal**, to directly test whether
   Sequential Learning can discover and converge on a genuinely better
   arm when there is one to find.
4. **Replace the synthetic experience log with real recorded
   executions**, removing the single largest threat to validity in both
   this campaign and its predecessor.
5. **Apply a multiple-comparisons correction** (Bonferroni or
   Benjamini-Hochberg) for any future campaign with more borderline
   p-values than this one.
