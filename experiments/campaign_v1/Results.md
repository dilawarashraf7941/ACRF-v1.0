# ACRF v1.0 Experimental Campaign — Results

**Campaign ID:** `campaign_v1`
**Executed via:** `experiments/campaign_v1/run_campaign.py` (execution-only; calls
only existing, unmodified `app/evaluation/{offline,experiments,ablation,statistics}`
classes — see that script's module docstring and `Threats_to_Validity.md`
for the full provenance/reproducibility statement).

**Reproducibility.** Synthetic experience log: 300 records, generator
seed `12345`. Every experiment/ablation: `random_seed=2024`,
`num_runs=30` (30 bootstrap resamples of the 300-record log, per
`app/evaluation/experiments/README.md`'s documented replay method).
Candidate action set: `LogicCritic`, `CodeCritic`, `FactCritic`,
`MetaCritic`. Re-running the script reproduces every number below
exactly (verified: source repository's record count was asserted
unchanged, `300`, both before and after the campaign).

Full-detail machine-readable exports: `results/core_experiments.json`,
`results/core_experiments.csv`, `results/ablations.json`,
`results/ablations.csv` (produced by the framework's own
`Exporter`/`AblationReportGenerator`, not hand-built).

---

## Table 1 — Overall Comparison

All ten requested experiments, with each candidate's paired comparison
against its natural reference (`Baseline` for experiments 2–6 and 10;
`Full ACRF` for the three ablations, 7–9).

| Experiment | Avg Reward | Avg Quality | Avg Latency | Avg Iterations | Match Rate | 95% CI (reward) | vs | p-value | Effect Size | Significant |
|---|---|---|---|---|---|---|---|---|---|---|
| 1. Baseline (HeuristicPolicy) | 0.7864 | 0.7768 | 1.0640 | 1.0096 | 0.3931 | [0.7652, 0.8120] | - | n/a | - | - |
| 2. LinUCB alpha=0 | 0.7864 | 0.7768 | 1.0640 | 1.0096 | 0.3931 | [0.7652, 0.8120] | Baseline | 1.0000 | 0.0000 | False |
| 3. LinUCB alpha=0.25 | 0.7864 | 0.7768 | 1.0640 | 1.0096 | 0.3931 | [0.7652, 0.8120] | Baseline | 1.0000 | 0.0000 | False |
| 4. LinUCB alpha=0.5 | 0.7864 | 0.7768 | 1.0640 | 1.0096 | 0.3931 | [0.7652, 0.8120] | Baseline | 1.0000 | 0.0000 | False |
| 5. LinUCB alpha=1.0 | 0.7864 | 0.7768 | 1.0640 | 1.0096 | 0.3931 | [0.7652, 0.8120] | Baseline | 1.0000 | 0.0000 | False |
| 6. LinUCB alpha=2.0 | 0.7864 | 0.7768 | 1.0640 | 1.0096 | 0.3931 | [0.7652, 0.8120] | Baseline | 1.0000 | 0.0000 | False |
| 10. Full ACRF (LinUCB alpha=1.0) | 0.7864 | 0.7768 | 1.0640 | 1.0096 | 0.3931 | [0.7652, 0.8120] | Baseline | 1.0000 | 0.0000 | False |
| 7. Random Critic | 0.6591 | 0.6959 | 1.3214 | 1.3125 | 0.2220 | [0.5885, 0.7254] | Full ACRF | <0.0001 | -3.9881 | **True** |
| 8. Reduced Context | 0.7864 | 0.7768 | 1.0640 | 1.0096 | 0.3931 | [0.7652, 0.8120] | Full ACRF | 1.0000 | 0.0000 | False |
| 9. Quality-only Reward | 0.7768 | 0.7768 | 1.0640 | 1.0096 | 0.3931 | [0.7621, 0.7888] | Full ACRF | 0.0003 | -0.7498 | **True** |

_(Raw table: `results/table1_overall_comparison.md`. Paired hypothesis
test auto-selected per `app/evaluation/statistics`'s decision logic —
every comparison above resolved to either `degenerate_zero_variance`
(rows 2–6, 8, 10 — every paired difference across all 30 runs was
exactly `0`) or `paired_t_test` (rows 7, 9).)_

## Table 2 — Ablation Comparison

Each ablation's candidate arm vs. **Full ACRF** (LinUCB, alpha=1.0,
standard `WeightedRewardStrategy`) as baseline — produced directly by
`AblationRunner`/`AblationReportGenerator`.

| Ablation | Baseline Reward (Full ACRF) | Candidate Reward | Reward Diff | Quality Diff | Latency Diff | Iteration Diff | Winner | p-value | Effect Size | Significant |
|---|---|---|---|---|---|---|---|---|---|---|
| 7. Random Critic | 0.7864 | 0.6591 | -0.1273 | -0.0810 | +0.2574 | +0.3030 | LinUCBPolicy | <0.0001 | -3.9881 | **True** |
| 8. Reduced Context | 0.7864 | 0.7864 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | tie | 1.0000 | 0.0000 | False |
| 9. Quality-only Reward | 0.7864 | 0.7768 | -0.0096 | +0.0000 | +0.0000 | +0.0000 | LinUCBPolicy | 0.0003 | -0.7498 | **True** |

_(Raw table: `results/table2_ablation_comparison.md`; full
narrative report: `results/ablations_report.md`.)_

## Table 3 — Alpha Sensitivity

The five requested LinUCB alpha values, each compared against the
Baseline (HeuristicPolicy).

| Alpha | Avg Reward | 95% CI (reward) | Avg Quality | Avg Latency | Avg Iterations | Match Rate | p-value vs Baseline | Effect Size |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.7864 | [0.7652, 0.8120] | 0.7768 | 1.0640 | 1.0096 | 0.3931 | 1.0000 | 0.0000 |
| 0.25 | 0.7864 | [0.7652, 0.8120] | 0.7768 | 1.0640 | 1.0096 | 0.3931 | 1.0000 | 0.0000 |
| 0.5 | 0.7864 | [0.7652, 0.8120] | 0.7768 | 1.0640 | 1.0096 | 0.3931 | 1.0000 | 0.0000 |
| 1.0 | 0.7864 | [0.7652, 0.8120] | 0.7768 | 1.0640 | 1.0096 | 0.3931 | 1.0000 | 0.0000 |
| 2.0 | 0.7864 | [0.7652, 0.8120] | 0.7768 | 1.0640 | 1.0096 | 0.3931 | 1.0000 | 0.0000 |

_(Raw table: `results/table3_alpha_sensitivity.md`.)_

---

## Figures

| File | Description |
|---|---|
| `figures/reward_curve.png` | Per-run `average_reward` across the 30 bootstrap runs, for Baseline, Full ACRF, Random Critic, and Quality-only Reward. Baseline and Full ACRF overlap exactly. |
| `figures/cumulative_reward.png` | Cumulative sum of the same per-run rewards. |
| `figures/reward_vs_alpha.png` | Average reward (with 95% CI error bars) at each of the 5 alpha values, against the Baseline reference line. |
| `figures/critic_selection_frequency.png` | Per-critic selection frequency for Baseline, Full ACRF, and Random Critic. |
| `figures/latency_comparison.png` | Average latency across all 10 experiments. |
| `figures/iterations_comparison.png` | Average iterations across all 10 experiments. |

---

## Headline Findings

1. **Baseline, all five LinUCB alpha values, and "Full ACRF" are
   numerically identical** across every reported metric — reward,
   quality, latency, iterations, match rate, and per-run values (30/30
   runs bit-identical). Every pairwise comparison against Baseline
   resolved to the `degenerate_zero_variance` test (p = 1.0000,
   effect size = 0.0000). **No statistically significant difference
   exists among experiments 1–6 and 10.** See `Discussion.md` for why.
2. **Random Critic Selection is the only candidate that differs from
   Full ACRF in critic-selection behavior**, and it is dramatically,
   statistically significantly worse: reward −0.1273 (p < 0.0001,
   Cohen's d = −3.99, a very large effect), driven by lower quality
   (−0.0810), higher latency (+0.2574), and more iterations (+0.3030).
3. **Reduced Context Features is indistinguishable from Full ACRF**
   (identical to the last decimal) — see `Discussion.md`.
4. **Quality-only Reward is statistically significantly different from
   the standard weighted reward** (p = 0.0003, d = −0.75, a medium-to-large
   effect) despite **identical critic-selection behavior and identical
   match rate** — the difference is entirely attributable to the reward
   *definition* itself (the weighted formula's completion bonus and
   modest penalties net positive relative to quality alone in this data).
