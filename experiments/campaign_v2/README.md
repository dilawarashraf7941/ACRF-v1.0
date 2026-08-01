# Campaign v2 — the experimental campaign reported in the manuscript

This is the campaign whose results `docs/Results.md`, `docs/Discussion.md`,
`docs/Conclusion.md`, and `docs/Threats_to_Validity.md` report. It
supersedes `experiments/campaign_v1/`, an earlier pilot run not
referenced by the manuscript.

## Running it

```bash
uv run python experiments/campaign_v2/run_campaign.py
```

Deterministic given its two fixed seeds (`DATA_SEED`, `CAMPAIGN_SEED`,
defined at the top of `run_campaign.py`); re-running overwrites
`results/` and `figures/` with identical output. Takes well under a
minute.

## Directory layout

| Path | Status | Content |
|---|---|---|
| `run_campaign.py` | current | The campaign script itself — dataset generation, all six policy configurations, statistics, tables, and figures |
| `results/` | **current** | Raw tables (`table*.md`), JSON/CSV exports, and per-configuration learning-curve data — the source of every number in `docs/Results.md` |
| `figures/` | **current** | The 9 PNGs referenced as Figures 3–11 in the manuscript |
| `archive/` | superseded | Pre-leakage-fix artifacts, kept only as an audit trail — see `archive/README.md` before opening anything inside it |

## Provenance

This campaign was run twice under the `campaign_v2` name:

1. **Original run** — against `app/evaluation/offline/replay.py`'s
   original offline context builder, which read `ExperienceRecord`
   fields unavailable at decision time (target leakage). Results from
   this run are archived (see above) and must not be cited.
2. **Revalidation run** (current) — after the leakage was fixed and the
   synthetic dataset generator in this file was minimally updated to
   supply the pre-decision fields (`task_type`, `planner_output`,
   `max_iterations`) the corrected context builder requires. This is
   the run `results/` and `figures/` — and the manuscript — reflect.

See the "Post-leakage-fix revalidation note" near the top of
`run_campaign.py` for the exact fields changed and why they don't
reintroduce leakage.
