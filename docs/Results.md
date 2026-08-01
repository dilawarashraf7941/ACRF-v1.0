# 5. Results

This section reports the results produced by the experimental procedure
described in Section 4. All values are drawn directly from the
generated campaign artifacts (`experiments/campaign_v2/results/`) and
are reported without interpretation; no causal or explanatory claims
are made in this section.

## 5.1 Overall Performance

Table 2 reports Average Reward, Average Quality, Average Latency,
Average Iterations, and Match Rate for the Heuristic Policy baseline,
the four Cold-Start LinUCB configurations, and the canonical
($\alpha=1.0$) Sequential Learning LinUCB configuration, each computed
over thirty bootstrap runs.

*Table 2: Overall comparison of all evaluated policies.*

| Policy | Avg Reward | Avg Quality | Avg Latency | Avg Iterations | Match Rate | 95% CI (Reward) |
|---|---|---|---|---|---|---|
| Heuristic Policy (Baseline) | 0.7951 | 0.7857 | 1.1136 | 0.9263 | 0.4017 | [0.7583, 0.8273] |
| Cold-Start LinUCB ($\alpha=0.25$) | 0.7951 | 0.7857 | 1.1136 | 0.9263 | 0.4017 | [0.7583, 0.8273] |
| Cold-Start LinUCB ($\alpha=0.5$) | 0.7951 | 0.7857 | 1.1136 | 0.9263 | 0.4017 | [0.7583, 0.8273] |
| Cold-Start LinUCB ($\alpha=1.0$) | 0.7951 | 0.7857 | 1.1136 | 0.9263 | 0.4017 | [0.7583, 0.8273] |
| Cold-Start LinUCB ($\alpha=2.0$) | 0.7951 | 0.7857 | 1.1136 | 0.9263 | 0.4017 | [0.7583, 0.8273] |
| Sequential Learning LinUCB ($\alpha=1.0$) | 0.7653 | 0.7525 | 1.1128 | 0.9994 | 0.0072 | [0.4740, 0.9616] |

The Heuristic Policy baseline and all four Cold-Start LinUCB
configurations report identical values for every metric in Table 2, to
the four decimal places shown. The Sequential Learning LinUCB
configuration reports a lower Average Reward (0.7653), lower Average
Quality (0.7525), an Average Latency essentially unchanged from (and
marginally lower than) the baseline (1.1128 vs. 1.1136), higher Average
Iterations (0.9994), and a substantially lower Match Rate (0.0072) than
the other five rows. Figure 9 presents Average Latency for every
evaluated configuration, including the ablation configurations reported
in Section 5.5.

## 5.2 Sequential Learning

Table 4 reports, for each of the four evaluated $\alpha$ values, the
Cold-Start and Sequential Learning LinUCB configurations' Average
Reward from the bootstrap procedure, together with the Sequential
Learning configuration's Convergence Step, Learning Rate estimate, and
Final Cumulative Regret computed from a single deterministic replay
pass (Section 4.5, Stage 4).

*Table 4: Sequential learning analysis across the evaluated $\alpha$ sweep.*

| $\alpha$ | Cold-Start Reward | Sequential Reward | Convergence Step | Learning Rate | Final Cum. Regret |
|---|---|---|---|---|---|
| 0.25 | 0.7951 | 0.7749 | 119 | -0.0002 | 31.4848 |
| 0.5 | 0.7951 | 0.7875 | 119 | -0.0002 | 31.4848 |
| 1.0 | 0.7951 | 0.7653 | 3 | -0.0187 | 0.2421 |
| 2.0 | 0.7951 | 0.5896 | 3 | -0.1416 | 1.5336 |

Table 5 reports the number of matched replay steps and Final Cumulative
Reward from the same single deterministic pass, for every evaluated
configuration.

*Table 5: Learning analysis summary (single-pass results) for every
evaluated configuration.*

| Experiment | Steps Matched | Avg Reward (single pass) | Final Cum. Reward |
|---|---|---|---|
| Heuristic Policy | 123 | 0.7949 | 97.7747 |
| Cold-Start LinUCB ($\alpha=0.25$-$2.0$) | 123 | 0.7949 | 97.7747 |
| Sequential Learning LinUCB ($\alpha=0.25$) | 123 | 0.7949 | 97.7747 |
| Sequential Learning LinUCB ($\alpha=0.5$) | 123 | 0.7949 | 97.7747 |
| Sequential Learning LinUCB ($\alpha=1.0$) | 4 | 0.9890 | 3.9560 |
| Sequential Learning LinUCB ($\alpha=2.0$) | 4 | 0.6661 | 2.6645 |

The four Cold-Start LinUCB rows report identical values in Table 5 and
are consolidated into one row above; the full, non-consolidated table
is provided as `table4_learning_analysis_summary.md`. The number of
Steps Matched for Sequential Learning LinUCB is 123 at both
$\alpha=0.25$ and $\alpha=0.5$ — identical to the 123 steps matched by
every Cold-Start LinUCB configuration and the Heuristic Policy — and 4
at both $\alpha=1.0$ and $\alpha=2.0$. Figure 3 presents Reward per
Step, Figure 4 presents Cumulative Reward, and Figure 5 presents Moving
Average Reward (ten-step trailing window) for the Heuristic Policy,
Cold-Start LinUCB ($\alpha=1.0$), Sequential Learning LinUCB
($\alpha=1.0$), and Random Critic configurations, over their respective
single deterministic replay passes.

## 5.3 Regret Analysis

Figure 6 presents Instantaneous Regret and Figure 7 presents Cumulative
Regret, both computed relative to the best reward observed within each
configuration's own single-pass replay sequence, for the same four
configurations as Figure 3-5. Final Cumulative Regret values for every
configuration are reported in Table 4 and Table 5, reproduced above:
31.4848 for the Heuristic Policy and every Cold-Start LinUCB
configuration; 31.4848, 31.4848, 0.2421, and 1.5336 for Sequential
Learning LinUCB at $\alpha=0.25$, $0.5$, $1.0$, and $2.0$ respectively;
22.9288 for Random Critic; 31.4848 for Reduced Context; and 26.5708 for
Quality-only Reward (the two ablation values are reported in Table 5's
full form, `table4_learning_analysis_summary.md`).

Figure 11 presents Moving Average Reward for the Heuristic Policy,
Cold-Start LinUCB ($\alpha=1.0$), Sequential Learning LinUCB
($\alpha=1.0$), and Random Critic configurations, each annotated with a
vertical marker at that configuration's Convergence Step. Convergence
Step values, reported in Table 4 and Table 5, are 119 for the Heuristic
Policy and every Cold-Start LinUCB configuration; 119, 119, 3, and 3
for Sequential Learning LinUCB at $\alpha=0.25$, $0.5$, $1.0$, and
$2.0$ respectively; and 62 for Random Critic.

## 5.4 Statistical Analysis

Table 6 reports the paired statistical comparison — automatically
selected test, $p$-value, 95% confidence interval for the mean
difference, Cohen's $d$ effect size, sample size, and significance at
$\alpha=0.05$ — for every comparison performed in this study (Section
4.5, Stage 7). Each comparison uses thirty paired per-run Average
Reward values.

*Table 6: Statistical comparison results for every evaluated pair.*

| Comparison | Test Used | $p$-value | 95% CI (Difference) | Effect Size ($d$) | $n$ | Significant |
|---|---|---|---|---|---|---|
| Cold-Start LinUCB ($\alpha=0.25$) vs. Heuristic | degenerate_zero_variance | 1.0000 | [0.0000, 0.0000] | 0.0000 | 30 | False |
| Cold-Start LinUCB ($\alpha=0.5$) vs. Heuristic | degenerate_zero_variance | 1.0000 | [0.0000, 0.0000] | 0.0000 | 30 | False |
| Cold-Start LinUCB ($\alpha=1.0$) vs. Heuristic | degenerate_zero_variance | 1.0000 | [0.0000, 0.0000] | 0.0000 | 30 | False |
| Cold-Start LinUCB ($\alpha=2.0$) vs. Heuristic | degenerate_zero_variance | 1.0000 | [0.0000, 0.0000] | 0.0000 | 30 | False |
| Sequential Learning ($\alpha=1.0$) vs. Heuristic | wilcoxon_signed_rank | 0.5561 | [-0.0811, 0.0215] | -0.2168 | 30 | False |
| Sequential vs. Cold-Start ($\alpha=0.25$) | wilcoxon_signed_rank | 0.3173 | [-0.0614, 0.0211] | -0.1826 | 30 | False |
| Sequential vs. Cold-Start ($\alpha=0.5$) | wilcoxon_signed_rank | 0.3173 | [-0.0230, 0.0079] | -0.1826 | 30 | False |
| Sequential vs. Cold-Start ($\alpha=1.0$) | wilcoxon_signed_rank | 0.5561 | [-0.0811, 0.0215] | -0.2168 | 30 | False |
| Sequential vs. Cold-Start ($\alpha=2.0$) | paired_t_test | <0.0001 | [-0.2467, -0.1643] | -1.8625 | 30 | True |
| Random Critic vs. Full ACRF | paired_t_test | <0.0001 | [-0.1118, -0.0952] | -4.6491 | 30 | True |
| Reduced Context vs. Full ACRF | degenerate_zero_variance | 1.0000 | [0.0000, 0.0000] | 0.0000 | 30 | False |
| Quality-only Reward vs. Full ACRF | paired_t_test | 0.0016 | [-0.0149, -0.0039] | -0.6372 | 30 | True |

The `degenerate_zero_variance` test outcome is recorded whenever every
one of the thirty paired differences is exactly zero. Confidence
intervals for the Sequential-vs-Heuristic and Sequential-vs-Cold-Start
($\alpha=1.0$) rows are identical, as both comparisons use the same
Sequential Learning LinUCB ($\alpha=1.0$) run data against reference
configurations that themselves report identical values (Table 2); the
same identity holds between the Sequential-vs-Cold-Start $\alpha=0.25$
and $\alpha=0.5$ rows, for the same reason (Section 5.2). Figure 10
presents the distribution of per-run Average Reward across the thirty
bootstrap runs, as a box plot, for the Heuristic Policy, Cold-Start
LinUCB ($\alpha=1.0$), Sequential Learning LinUCB ($\alpha=1.0$), and
Random Critic configurations.

## 5.5 Ablation Results

Table 3 reports the ablation study results: each candidate
configuration's Average Reward, together with its difference from the
Full ACRF baseline (Cold-Start LinUCB, $\alpha=1.0$) in Average Reward,
Average Quality, Average Latency, and Average Iterations, its Match
Rate, and the designated winner.

*Table 3: Ablation study results (each candidate vs. Full ACRF).*

| Ablation | Baseline Reward | Candidate Reward | Reward Diff | Quality Diff | Latency Diff | Iteration Diff | Match Rate | Winner |
|---|---|---|---|---|---|---|---|---|
| Random Critic | 0.7951 | 0.6916 | -0.1035 | -0.0722 | +0.1936 | +0.2584 | 0.2280 | LinUCBPolicy |
| Reduced Context | 0.7951 | 0.7951 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0.4017 | tie |
| Quality-only Reward | 0.7951 | 0.7857 | -0.0094 | +0.0000 | +0.0000 | +0.0000 | 0.4017 | LinUCBPolicy |

Figure 8 presents per-critic selection frequency for the Heuristic
Policy, Full ACRF, Sequential Learning LinUCB ($\alpha=1.0$), and Random
Critic configurations. Corresponding statistical comparison values for
each ablation are reported in Table 6 (Section 5.4).

The alpha sweep results are reported in Table 2 (Cold-Start LinUCB,
$\alpha \in \{0.25, 0.5, 1.0, 2.0\}$, each compared against the
Heuristic Policy baseline) and in Table 4 (Sequential Learning LinUCB
at the same four $\alpha$ values, each compared against Cold-Start
LinUCB at the matching $\alpha$ value). Across the four Cold-Start
LinUCB configurations, Average Reward, Average Quality, Average
Latency, Average Iterations, Match Rate, and 95% confidence interval
are identical at every tested $\alpha$ value (Table 2). Across the four
Sequential Learning LinUCB configurations, Average Reward ranges from
0.5896 ($\alpha=2.0$) to 0.7875 ($\alpha=0.5$); Convergence Step takes
one of two values, 119 ($\alpha=0.25$ and $\alpha=0.5$) or 3
($\alpha=1.0$ and $\alpha=2.0$); and Final Cumulative Regret likewise
takes one of two values, 31.4848 ($\alpha=0.25$ and $\alpha=0.5$) or
0.2421/1.5336 ($\alpha=1.0$/$\alpha=2.0$ respectively) (Table 4).

## Chapter Summary

This section reported the results of the experimental procedure defined
in Section 4, without interpretation. Section 5.1 reported that the
Heuristic Policy and all four Cold-Start LinUCB configurations recorded
identical Average Reward, Average Quality, Average Latency, Average
Iterations, Match Rate, and confidence intervals, while the canonical
Sequential Learning LinUCB configuration recorded different values on
every one of these metrics. Section 5.2 and 5.3 reported per-step
reward, cumulative reward, moving average, instantaneous and cumulative
regret, and convergence-step values for the sequential learning
configurations across the evaluated $\alpha$ sweep, together with the
corresponding single-pass values for every other configuration. Section
5.4 reported the automatically selected statistical test, $p$-value,
95% confidence interval, effect size, and significance outcome for
twelve paired comparisons, of which three were reported as
statistically significant at $\alpha=0.05$ and nine were reported as
not significant. Section 5.5 reported the three
ablation comparisons and the full alpha-sweep results for both
Cold-Start and Sequential Learning LinUCB configurations. Complete,
unabridged tabular data corresponding to every table in this section is
available in `experiments/campaign_v2/results/`.
