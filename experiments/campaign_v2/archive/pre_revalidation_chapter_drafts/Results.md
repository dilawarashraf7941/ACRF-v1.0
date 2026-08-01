> **ARCHIVED — SUPERSEDED, DO NOT CITE.** Generated before the
> target-leakage fix in the offline replay context builder; every
> number below is invalid. See `../README.md` and the current
> `docs/Results.md` for the revalidated results.

# ACRF v1.0 Final Experimental Campaign — Results

**Campaign ID:** `campaign_v2`
**Executed via:** `experiments/campaign_v2/run_campaign.py` — execution-only; calls
only existing, unmodified ACRF classes (`ExperienceRecord`,
`InMemoryExperienceRepository`, `ExperimentRunner`, `AblationRunner`,
`ReplayEngine` — both `.replay()` and `.replay_with_learning()` —,
`OfflineEvaluator`, `Benchmark`, `app.evaluation.statistics.Analyzer`,
`app.evaluation.learning_analysis.LearningAnalyzer`, and their
exporters). See that script's module docstring and
`Threats_to_Validity.md` for the full provenance/reproducibility
statement.

## How to reproduce

```bash
cd <repo root>
python experiments/campaign_v2/run_campaign.py
```

No other setup is required beyond the project's normal dependencies
(`langgraph`, `pydantic`, `numpy`, `scipy`) plus `matplotlib` (used only
by this script, for figure generation — not a framework dependency).
Re-running reproduces every number and figure in this report exactly:
the synthetic experience log is generated with a fixed seed
(`DATA_SEED = 12345`, identical generator to the prior `campaign_v1`,
for continuity) and every bootstrap-based experiment/ablation uses a
fixed `CAMPAIGN_SEED = 2024`. Total runtime: **28.06 seconds** (see
`results/runtime.txt`).

## Experiments executed

1. **Heuristic Policy** (baseline)
2. **Cold-Start LinUCB** — alpha in `{0.25, 0.5, 1.0, 2.0}` (the
   framework's own configured alpha sweep,
   `app.evaluation.ablation.DEFAULT_ALPHA_SWEEP`)
3. **Sequential Learning LinUCB** — same 4 alpha values, via
   `ReplayEngine.replay_with_learning()`
4. **Random Critic** (ablation, vs. Full ACRF = Cold-Start LinUCB alpha=1.0)
5. **Reduced Context** (ablation, vs. Full ACRF)
6. **Quality-only Reward** (ablation, vs. Full ACRF)

Every bootstrap-based statistic below is computed over **30 independent
bootstrap resamples** (`NUM_RUNS = 30`) of a **300-record** synthetic
experience log (`NUM_RECORDS = 300`). Every "raw curve" statistic
(reward per step, cumulative reward/regret, moving average,
convergence step, learning rate) is computed from **one deterministic
pass** over the full, non-resampled log — mirroring how
`ExperimentRunner` itself treats `num_runs == 1`.

---

## Table 1 — Overall Comparison of All Policies

| Policy | Avg Reward | Avg Quality | Avg Latency | Avg Iterations | Match Rate | 95% CI (reward) | p-value | Effect Size | Significant |
|---|---|---|---|---|---|---|---|---|---|
| 1. Heuristic Policy (Baseline) | 0.7864 | 0.7768 | 1.0640 | 1.0096 | 0.3931 | [0.7652, 0.8120] | n/a | - | - |
| 2. Cold-Start LinUCB (alpha=0.25) | 0.7864 | 0.7768 | 1.0640 | 1.0096 | 0.3931 | [0.7652, 0.8120] | 1.0000 | 0.0000 | False |
| 2. Cold-Start LinUCB (alpha=0.5) | 0.7864 | 0.7768 | 1.0640 | 1.0096 | 0.3931 | [0.7652, 0.8120] | 1.0000 | 0.0000 | False |
| 2. Cold-Start LinUCB (alpha=1.0) | 0.7864 | 0.7768 | 1.0640 | 1.0096 | 0.3931 | [0.7652, 0.8120] | 1.0000 | 0.0000 | False |
| 2. Cold-Start LinUCB (alpha=2.0) | 0.7864 | 0.7768 | 1.0640 | 1.0096 | 0.3931 | [0.7652, 0.8120] | 1.0000 | 0.0000 | False |
| 3. Sequential Learning LinUCB (alpha=1.0, canonical) | 0.5559 | 0.6644 | 1.4052 | 1.6354 | 0.0823 | [0.4658, 0.6226] | <0.0001 | -5.0725 | **True** |

_(Raw: `results/table1_overall_comparison.md`.)_

## Table 2 — Ablation Study

| Ablation | Baseline Reward (Full ACRF) | Candidate Reward | Reward Diff | Quality Diff | Latency Diff | Iteration Diff | Match Rate | Winner | p-value | Effect Size | Significant |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4. Random Critic | 0.7864 | 0.6591 | -0.1273 | -0.0810 | +0.2574 | +0.3030 | 0.2220 | LinUCBPolicy | <0.0001 | -3.9881 | **True** |
| 5. Reduced Context Ablation | 0.7864 | 0.7864 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0.3931 | tie | 1.0000 | 0.0000 | False |
| 6. Quality-only Reward Ablation | 0.7864 | 0.7768 | -0.0096 | +0.0000 | +0.0000 | +0.0000 | 0.3931 | LinUCBPolicy | 0.0003 | -0.7498 | **True** |

_(Raw: `results/table2_ablation_study.md`; full narrative: `results/ablations_report.md`.)_

## Table 3 — Sequential Learning Analysis (Cold-Start vs. Sequential, same alpha)

| Alpha | Cold-Start Reward | Sequential Reward | Reward Diff | p-value | Effect Size | Significant | Convergence Step | Learning Rate | Final Cum. Regret |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 0.7864 | 0.7735 | -0.0129 | 0.0003 | -0.7488 | **True** | 119 | -0.0010 | 41.5824 |
| 0.5 | 0.7864 | 0.7028 | -0.0836 | <0.0001 | -1.7314 | **True** | 39 | +0.0057 | 14.0930 |
| 1.0 | 0.7864 | 0.5559 | -0.2304 | <0.0001 | -5.0725 | **True** | 23 | +0.0025 | 6.4241 |
| 2.0 | 0.7864 | 0.5444 | -0.2420 | <0.0001 | -5.5939 | **True** | 23 | -0.0106 | 8.8503 |

_(Raw: `results/table3_sequential_learning_analysis.md`.)_

## Table 4 — Learning Analysis Summary (raw single-pass curve, every experiment)

| Experiment | Steps Matched | Avg Reward | Final Cum. Reward | Final Cum. Regret | Convergence Step | Learning Rate |
|---|---|---|---|---|---|---|
| 1. Heuristic Policy | 120 | 0.7909 | 94.9107 | 39.2859 | 118 | -0.0003 |
| 2. Cold-Start LinUCB (alpha=0.25) | 120 | 0.7909 | 94.9107 | 39.2859 | 118 | -0.0003 |
| 2. Cold-Start LinUCB (alpha=0.5) | 120 | 0.7909 | 94.9107 | 39.2859 | 118 | -0.0003 |
| 2. Cold-Start LinUCB (alpha=1.0) | 120 | 0.7909 | 94.9107 | 39.2859 | 118 | -0.0003 |
| 2. Cold-Start LinUCB (alpha=2.0) | 120 | 0.7909 | 94.9107 | 39.2859 | 118 | -0.0003 |
| 3. Sequential Learning LinUCB (alpha=0.25) | 122 | 0.7775 | 94.8509 | 41.5824 | 119 | -0.0010 |
| 3. Sequential Learning LinUCB (alpha=0.5) | 41 | 0.6030 | 24.7246 | 14.0930 | 39 | +0.0057 |
| 3. Sequential Learning LinUCB (alpha=1.0) | 24 | 0.6189 | 14.8545 | 6.4241 | 23 | +0.0025 |
| 3. Sequential Learning LinUCB (alpha=2.0) | 24 | 0.5906 | 14.1734 | 8.8503 | 23 | -0.0106 |
| 4. Random Critic | 74 | 0.6480 | 47.9498 | 26.1276 | 73 | -0.0006 |
| 5. Reduced Context | 120 | 0.7909 | 94.9107 | 39.2859 | 118 | -0.0003 |
| 6. Quality-only Reward | 120 | 0.7774 | 93.2932 | 26.7068 | 117 | -0.0001 |

_(Raw: `results/table4_learning_analysis_summary.md`; full per-step CSV/JSON for the 4 key series: `results/learning_curve_*.csv`/`.json`; all 12 curves: `results/learning_curves_all.json`.)_

---

## Figures

| File | Description |
|---|---|
| `figures/reward_curve.png` | Per-step reward for Heuristic, Cold-Start LinUCB, Sequential Learning LinUCB, Random Critic (alpha=1.0 for the LinUCB variants). |
| `figures/cumulative_reward.png` | Cumulative sum of the same per-step rewards. |
| `figures/moving_average_reward.png` | Trailing moving average (window=10) of the same series. |
| `figures/instantaneous_regret.png` | Per-step regret relative to the best reward observed in each series. |
| `figures/cumulative_regret.png` | Running sum of instantaneous regret. |
| `figures/critic_selection_distribution.png` | Per-critic selection frequency (bootstrap-aggregated) for the same 4 policies. |
| `figures/latency_comparison.png` | Average latency across all 12 bootstrap-evaluated experiments. |
| `figures/reward_distribution.png` | Box plot of per-run average reward across the 30 bootstrap runs, for the same 4 key policies. |
| `figures/convergence_analysis.png` | Moving average reward with a vertical dashed line at each series' convergence step. |

---

## Headline Findings

1. **Heuristic Policy, all four Cold-Start LinUCB alpha values, and Reduced Context are numerically identical** across every metric (30/30 bootstrap runs bit-identical; every comparison against Baseline resolves to `degenerate_zero_variance`, p=1.0). Mechanically explained in `Discussion.md` — this is the same, now-repeatedly-confirmed cold-start convergence finding from prior campaigns.
2. **Sequential Learning LinUCB performs significantly *worse* than Cold-Start LinUCB at every tested alpha**, and the gap widens sharply as alpha increases: -0.0129 reward at alpha=0.25 (p=0.0003) up to -0.2420 at alpha=2.0 (p<0.0001, d=-5.59, an extremely large effect). **This is not an improvement — it is a clear, statistically significant regression**, and is reported here exactly as observed, per the instruction not to exaggerate.
3. The mechanism is directly visible in Table 3/4: **higher alpha causes earlier, permanent drift away from the best-performing default critic** (convergence step drops from 119 at alpha=0.25 to 23 at alpha=1.0-2.0; matched steps drop from 122 to 24), after which the "replay method"'s match-gated training never gets a chance to correct the switch, since the abandoned arm is never selected — and hence never updated — again.
4. **Random Critic Selection** remains significantly worse than Full ACRF (reward -0.1273, p<0.0001, d=-3.99) — and, notably, **still outperforms Sequential Learning LinUCB at alpha>=0.5** (Random Critic median bootstrap reward ~0.65 vs. Sequential Learning's ~0.55-0.60; see `figures/reward_distribution.png`), underscoring how costly the early, irreversible arm-switch is in this setup.
5. **Reduced Context Features remains indistinguishable from Full ACRF** (identical to the last decimal) — the same cold-start-invariance mechanism as prior campaigns.
6. **Quality-only Reward** remains significantly different from the standard weighted reward (p=0.0003, d=-0.75) despite identical critic-selection behavior — a reward-*definition* effect, unchanged from `campaign_v1`.
