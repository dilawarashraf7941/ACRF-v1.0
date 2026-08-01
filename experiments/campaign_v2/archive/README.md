# Archive: pre-revalidation campaign_v2 artifacts

**These outputs are superseded and must not be cited.** They were
produced by `run_campaign.py` *before* a target-leakage defect in
`app/evaluation/offline/replay.py`'s offline context construction was
fixed, and before the synthetic dataset generator was updated to supply
the pre-decision fields the corrected context builder requires.

Under the leaky context, the offline replay policy could see features
derived from an episode's own outcome — invalidating any result
downstream of it, most severely the Sequential Replay Learning findings
(Discussion §6.2 of the manuscript). The campaign was re-run end to end
against the corrected code, and every number in `docs/Results.md`,
`docs/Discussion.md`, `docs/Conclusion.md`, and
`docs/Threats_to_Validity.md` reflects that revalidated run.

## What's here

| Path | Content |
|---|---|
| `pre_revalidation_results/` | Raw tables, JSON/CSV exports, and learning-curve data from the pre-fix run |
| `pre_revalidation_figures/` | The 9 PNG figures generated from that same pre-fix run |
| `pre_revalidation_chapter_drafts/` | Stray, never-updated draft copies of `Discussion.md`, `Results.md`, and `Threats_to_Validity.md`, generated at some point during authoring and left uncorrected — these still contain the pre-fix numbers (e.g. Sequential Learning LinUCB reward 0.5559, `p<0.0001`) and must never be read as current |

## Where the current, valid outputs are

- Current raw campaign outputs: `experiments/campaign_v2/results/` and
  `experiments/campaign_v2/figures/`.
- Current manuscript chapters: `docs/Results.md`, `docs/Discussion.md`,
  `docs/Conclusion.md`, `docs/Threats_to_Validity.md`.
- The dataset-generator fix itself and its rationale: see the
  "Post-leakage-fix revalidation note" in `run_campaign.py`'s module
  docstring.

This archive is kept for audit-trail purposes only — to make the
before/after effect of the leakage fix independently checkable — not as
an alternative source of results.
