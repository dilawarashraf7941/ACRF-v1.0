# Manuscript — Adaptive Critic Routing Framework (ACRF) v1.0

Reading order and file map for the paper. Each chapter is a standalone
Markdown file; there is no single concatenated document.

| # | Chapter | File |
|---|---|---|
| — | Abstract & Keywords | [`Abstract.md`](Abstract.md) |
| 1 | Introduction | [`Introduction.md`](Introduction.md) |
| 2 | Related Work | [`Related_Work.md`](Related_Work.md) |
| 3 | Methodology | [`Methodology.md`](Methodology.md) |
| 4 | Experimental Setup | [`Experimental_Setup.md`](Experimental_Setup.md) |
| 5 | Results | [`Results.md`](Results.md) |
| 6 | Discussion | [`Discussion.md`](Discussion.md) |
| 7 | Threats to Validity | [`Threats_to_Validity.md`](Threats_to_Validity.md) |
| 8 | Conclusion | [`Conclusion.md`](Conclusion.md) |
| — | References | [`References.md`](References.md) |

## Data provenance

Every number in Sections 5, 6, 7, and 8.3 traces to
[`experiments/campaign_v2/results/`](../experiments/campaign_v2/results/)
and [`experiments/campaign_v2/figures/`](../experiments/campaign_v2/figures/),
reproducible by running `experiments/campaign_v2/run_campaign.py` (see
the root `README.md`, "Reproducing the experiments"). Do not confuse
these with the superseded artifacts under
`experiments/campaign_v2/archive/`.

## Completeness

Every figure, algorithm, and table the manuscript references now
exists. Algorithm 1 (Section 3.1), Figure 1 (Section 3.1), Table 1
(Section 3.3), Figure 2 (Section 3.7), and Algorithm 2 (Section 4.5)
were the last five items resolved, each derived directly from the
implementation (`app/graph/`, `app/policy_engine/scorer.py`) or from
`experiments/campaign_v2/run_campaign.py` and its already-generated
outputs — no new experiments were run to produce them. Figures 3-11 and
Tables 2-6 (Sections 5-8) were already in place, generated directly
from `experiments/campaign_v2/`.
