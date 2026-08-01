# 4. Experimental Setup

This section describes the experimental configuration used to evaluate
the components of ACRF v1.0 described in Section 3. All experiments
were conducted entirely within the offline evaluation pipeline
(Section 3.6); no experiment invoked the live execution graph, a
language model, or any external service. This section describes only
how the experiments were conducted; results and their interpretation
are reported separately.

## 4.1 Research Questions

The experimental campaign was designed to answer five research
questions, each directly supported by one or more of the completed
experiments described in Section 4.3.

**RQ1.** Can policy-guided adaptive critic routing outperform
non-adaptive routing strategies — a deterministic heuristic and
uniformly random critic selection? Addressed by comparing the
Heuristic Policy and Random Critic baselines against the Cold-Start and
Sequential Learning LinUCB configurations (Table 2, Table 3).

**RQ2.** How does sequential replay learning affect policy adaptation
relative to an untrained (cold-start) instance of the same policy?
Addressed by comparing Cold-Start LinUCB and Sequential Learning LinUCB
at matched exploration-coefficient values, and by the per-step
learning-curve metrics computed for each (Table 4, Table 5).

**RQ3.** What is the effect of the exploration coefficient $\alpha$ on
routing behavior, under both the standard (non-learning) and sequential
learning evaluation modes? Addressed by evaluating both modes across
the framework's configured $\alpha$ sweep (Table 4).

**RQ4.** How sensitive is ACRF's LinUCB policy to a reduction in the
number of context features available at decision time? Addressed by
the Reduced Context ablation (Table 3).

**RQ5.** How robust is the measured policy value to the definition of
the reward function? Addressed by the Quality-only Reward ablation,
which substitutes an alternative reward strategy while holding the
policy and context fixed (Table 3).

## 4.2 Experimental Environment

**Programming language.** All framework and experiment code is written
in Python, targeting Python 3.12.

**Major libraries.** NumPy provides the array and linear-algebra
operations underlying the LinUCB policy's per-arm statistics (design
matrix, inverse, and response vector) and its Sherman-Morrison update.
SciPy provides the statistical primitives used by the statistical
analysis component (Section 3.6): the Shapiro-Wilk normality test, the
paired $t$-test, the Wilcoxon signed-rank test, and the Student's
$t$-distribution quantile function used to compute confidence intervals
for paired mean differences. LangGraph defines the state-graph
abstraction (`StateGraph`) that the architecture described in Section
3.1 is built on; because every experiment in this study operates on
already-recorded experience data through the offline evaluation
pipeline, no experiment executes the compiled graph directly. Pydantic
(schema version 2) is used throughout for every structured data model
in the framework and in the evaluation pipeline — experience records,
context vectors, policy decisions, replay results, and every model
described in Section 3 — providing validated, immutable data exchange
between components. Matplotlib was used only by the experiment script
to render figures from already-computed tabular data; it is not a
dependency of the framework itself.

**Operating assumptions.**

- *Offline evaluation.* No experiment invokes a language model, the
  compiled execution graph, or any live external service. Every
  reported quantity is derived from the offline replay of previously
  recorded experience data, as described in Section 3.4 and Section
  3.6.
- *Synthetic dataset.* Because ACRF v1.0 does not perform live
  execution, no naturally occurring experience log exists. All
  experiments in this study were therefore conducted against a
  synthetically generated set of experience records, described in
  Section 4.2.1 below. This is an explicit property of the experimental
  setup, not a property of the framework, which is agnostic to how
  experience records are produced.
- *Sequential replay.* Wherever an experiment is described as using
  sequential replay learning, this refers exclusively to the
  `replay_with_learning()` evaluation mode described in Section 3.4;
  every other experiment uses the standard, non-learning replay mode,
  which never modifies policy state.

### 4.2.1 Synthetic Dataset

The experience log used throughout this study consists of 300
synthetically generated experience records, produced by a fixed,
seeded generator (seed 12345) so that the dataset itself is
reproducible independently of any experiment run against it. Each
record is drawn from one of five fixed archetypes, each specifying a
selection weight, a set of selected critic identifiers, and Gaussian
parameters (mean and standard deviation) for the record's quality
score, latency, and iteration count:

- 40% of records: a single `CodeCritic` selection, quality
  $\mathcal{N}(0.78, 0.08)$, latency $\mathcal{N}(1.1, 0.25)$ seconds,
  iteration count $\mathcal{N}(0.8, 1.0)$.
- 20% of records: a single `LogicCritic` selection, quality
  $\mathcal{N}(0.68, 0.09)$, latency $\mathcal{N}(1.4, 0.30)$ seconds,
  iteration count $\mathcal{N}(1.3, 1.0)$.
- 15% of records: a single `FactCritic` selection, quality
  $\mathcal{N}(0.60, 0.10)$, latency $\mathcal{N}(1.6, 0.30)$ seconds,
  iteration count $\mathcal{N}(1.6, 1.0)$.
- 10% of records: a single `MetaCritic` selection, quality
  $\mathcal{N}(0.58, 0.12)$, latency $\mathcal{N}(1.8, 0.35)$ seconds,
  iteration count $\mathcal{N}(2.0, 1.0)$.
- 15% of records: a two-critic selection (`FactCritic`/`MetaCritic` or
  `LogicCritic`/`MetaCritic`, chosen uniformly at random per record),
  quality $\mathcal{N}(0.50, 0.13)$, latency $\mathcal{N}(2.2, 0.40)$
  seconds, iteration count $\mathcal{N}(2.8, 1.0)$.

For every record, quality is additionally clipped to $[0,1]$, latency
is bounded below by 0.2 seconds, and iteration count is rounded to the
nearest integer and clipped to $[0,6]$. Each critic's individual score
is drawn as the record's quality value perturbed by independent
Gaussian noise ($\sigma = 0.05$), then clipped to $[0,1]$. Terminal
execution status is set to "completed" with probability 0.92 and
"failed" with probability 0.08, independently per record. Record
timestamps advance monotonically from a fixed start date by a uniformly
random number of minutes (1-30) per record. All four built-in critic
identifiers (`LogicCritic`, `CodeCritic`, `FactCritic`, `MetaCritic`)
constitute the fixed candidate action set presented to every policy
throughout this study.

Each archetype additionally fixes two pre-decision properties the
offline context builder (Section 3.2) reads: a `task_type` label —
`"code"`, `"reasoning"`, `"research"`, `"escalation"`, and
`"multi_domain"` for the five archetypes above, in the same order —
and a mean decomposition length (`plan_steps_mean`, one of $1.5$,
$2.5$, $2.0$, $3.5$, $4.0$ respectively) from which each record's
actual decomposition length is drawn via its own independent Gaussian
draw ($\sigma=1.0$, rounded and clipped to $[0,6]$), never derived from
the record's quality, latency, iteration count, or status. Every record
additionally carries a fixed iteration budget, `max_iterations = 5`,
constant across the dataset. Only the `"code"` archetype's `task_type`
equals `"code"`; combined with the decomposition-length draw and the
fixed iteration budget, this generation procedure produces exactly
eleven distinct offline context vectors across the 300 stored records
under the context builder described in Section 3.2 — two possible
values of `is_code_task` crossed with the range of `plan_complexity`
values the decomposition-length draws realize, since `max_iterations`
and `has_task_type` are constant for every record.

## 4.3 Baselines

Six policy configurations were evaluated.

**Heuristic Policy.** The Heuristic Policy described in Section 3.3-A,
evaluated via standard (non-learning) replay. This configuration
constitutes the deterministic-routing baseline referenced in RQ1.

**Cold-Start LinUCB.** The LinUCB Policy described in Section 3.3-B,
freshly constructed (all per-arm statistics at their initial values)
for every replay, evaluated via standard replay at each of four
exploration-coefficient values, $\alpha \in \{0.25, 0.5, 1.0, 2.0\}$ —
the framework's configured default sweep. No policy instance is reused
or updated across the runs that make up this configuration's reported
statistics.

**Sequential Learning LinUCB.** The same LinUCB Policy and the same
four $\alpha$ values, evaluated via the sequential replay learning mode
described in Section 3.4. A freshly constructed (untrained) policy
instance is used for each independent run, so that this configuration
isolates the effect of training that occurs *within* a single replay
pass, as distinct from any effect of reusing a policy instance across
runs.

**Random Critic.** A policy that selects uniformly at random from the
candidate action set, implemented as an ablation of the framework's
default policy-construction mechanism (Section 3.6) and evaluated via
standard replay. This configuration is compared against Cold-Start
LinUCB at $\alpha=1.0$ as its reference baseline.

**Reduced Context.** The LinUCB Policy at $\alpha=1.0$, wrapped so that
only the first 50% of the context vector's named features (by the
vector's defined feature ordering, Section 3.2) are presented to the
policy at decision time; all other aspects of the configuration are
identical to Cold-Start LinUCB at $\alpha=1.0$, which serves as its
reference baseline.

**Quality-only Reward.** The LinUCB Policy at $\alpha=1.0$, with the
default weighted reward strategy (Section 3.5) replaced by an
alternative strategy that reports the record's quality score alone as
reward, with all other aspects of the configuration identical to
Cold-Start LinUCB at $\alpha=1.0$, which serves as its reference
baseline.

## 4.4 Evaluation Metrics

The following metrics, all computed by the components described in
Section 3.6 and Section 3.7, were recorded for every configuration in
Section 4.3.

- **Average Reward** — the mean reward over a policy's matched replay
  steps (Section 3.6), reported per bootstrap run and, in aggregate,
  averaged across runs.
- **Cumulative Reward** — the running sum of per-step reward
  (equation 12) over a single completed replay pass.
- **Reward per Step** — the raw, index-aligned sequence of per-step
  reward values from which Cumulative Reward, Moving Average Reward,
  and regret are derived.
- **Quality** — the mean of the aggregated quality score across a
  policy's matched replay steps.
- **Latency** — the mean of the recorded latency across a policy's
  matched replay steps.
- **Iterations** — the mean of the recorded iteration count across a
  policy's matched replay steps.
- **Match Rate** — the proportion of stored experience records for
  which a policy's live selection matched the historically recorded
  selection (Section 3.4), and which therefore contributed to every
  other metric in this list.
- **Moving Average Reward** — the trailing moving-average series
  (equation 14) computed with the framework's default window size of
  ten steps.
- **Cumulative Regret** — the running sum of hindsight instantaneous
  regret (equation 13) over a single completed replay pass.
- **Convergence Step** — the earliest step index from which the moving
  average series (above) remains within the framework's default
  tolerance band (5% of the series' observed range) of its own final
  value for the remainder of the pass.
- **Effect Size** — Cohen's $d$ for paired samples (equation 11),
  reported for every baseline-versus-candidate comparison.
- **Confidence Interval** — reported at the 95% level throughout; the
  empirical percentile interval (Section 3.6) is used for a single
  configuration's own reward distribution across bootstrap runs, and
  the $t$-distribution interval (equation 10) is used for the mean
  difference between two paired configurations.

Average Reward, Quality, Latency, Iterations, Match Rate, and Effect
Size/Confidence Interval were computed from thirty independent
bootstrap runs per configuration (Section 4.5). Reward per Step,
Cumulative Reward, Moving Average Reward, Cumulative Regret, and
Convergence Step were computed from a single, deterministic replay
pass over the complete (non-resampled) dataset for each configuration.

## 4.5 Experimental Procedure

Experiments were executed by a single script that performs the
following sequence of stages in order. Algorithm 2 summarizes this
sequence; each stage is described below.

```
Algorithm 2: Experimental campaign procedure (experiments/campaign_v2/run_campaign.py)

Input:  DATA_SEED, CAMPAIGN_SEED, NUM_RUNS = 30, NUM_RECORDS = 300,
        ALPHAS = (0.25, 0.5, 1.0, 2.0)
Output: 4 summary tables, 9 figures, JSON/CSV/Markdown exports

 1: repository ← generate_repository(DATA_SEED, NUM_RECORDS)              ▷ Stage 1
 2: dumps_before ← { r.experience_id: r.model_dump() for r in repository.list() }
 3:
 4: core_results ← run_core_bootstrap_experiments(repository)             ▷ Stage 2
        ▷ for policy in {HeuristicPolicy, LinUCBPolicy(α) for α in ALPHAS}:
        ▷     ExperimentRunner(repository).run(policy, num_runs=NUM_RUNS, seed=CAMPAIGN_SEED)
 5:
 6: sequential_results ← run_sequential_learning_experiments(repository)  ▷ Stage 3
        ▷ for α in ALPHAS:
        ▷     run_sequential_learning_bootstrap(repository, α)              (uses Stage 4's
        ▷         resamples repository NUM_RUNS times (BootstrapExperienceRepository);        bootstrap
        ▷         each run: fresh LinUCBPolicy(α), ReplayEngine.replay_with_learning()         procedure)
        ▷         aggregate the NUM_RUNS run-level ReplayResults into one ExperimentResult
 7:
 8: ablation_results ← run_ablations(repository)                          ▷ Stage 5
        ▷ AblationRunner(repository).run(config, num_runs=NUM_RUNS, seed=CAMPAIGN_SEED)
        ▷     for config in {random_critic, reduced_context, quality_only_reward}
 9: ablation_candidates ← replay_ablation_candidate_arms(repository)      ▷ Stage 5 (candidate arms,
        ▷ ExperimentRunner with each ablation's policy_factory/reward_calculator                 independently)
10:
11: raw_curves ← build_raw_curves(repository)                             ▷ Stage 6
        ▷ for each configuration: one deterministic .replay() or .replay_with_learning()
        ▷     pass over the full, non-resampled repository, then LearningAnalyzer().analyze(steps)
12:
13: comparisons ← compute_statistics(core_results, sequential_results,    ▷ Stage 7
                                       ablation_candidates)
        ▷ Analyzer().compare_experiments(baseline, candidate, metric="average_reward")
        ▷     for every {Cold-Start, Sequential Learning} × ALPHAS vs. Heuristic;
        ▷     Sequential vs. Cold-Start at matched α; each ablation vs. Full ACRF
14:
15: assert repository.count() = NUM_RECORDS                               ▷ Stage 7, integrity check
16: assert { r.experience_id: r.model_dump() for r in repository.list() } = dumps_before
17:
18: table1..4 ← build_table1(...), build_table2(...), build_table3(...), build_table4(...)  ▷ Output
19: generate_figures(raw_curves, core_results, sequential_results, ablation_candidates)      generation
20: write table1..4, figures, and JSON/CSV/Markdown exports to experiments/campaign_v2/{results,figures}/
```

*Algorithm 2: Experimental campaign procedure, transcribed directly
from `experiments/campaign_v2/run_campaign.py`'s `__main__` block and
the seven stage-numbered functions it calls, in the exact order they
execute. Every function name above is the actual function defined in
that file; none is abbreviated or renamed for presentation.*

**Stage 1 — Dataset generation.** The synthetic experience log
described in Section 4.2.1 is generated once, using its fixed seed, and
held fixed for every subsequent stage. A snapshot of every record's
serialized content is retained at this point for the integrity check
performed in Stage 7.

**Stage 2 — Replay.** The Heuristic Policy baseline and each Cold-Start
LinUCB configuration are evaluated using thirty independent bootstrap
runs each (Stage 4 defines this bootstrap procedure); each run replays
a freshly constructed policy instance via the standard, non-learning
replay mode against a resampled copy of the dataset.

**Stage 3 — Sequential replay.** Each Sequential Learning LinUCB
configuration is evaluated using the same bootstrap procedure, with two
differences from Stage 2: a freshly constructed policy instance is
replayed via the sequential replay learning mode (so its state evolves
within each individual run), and the per-run aggregation is performed
by dedicated procedure-level code, since sequential replay learning has
no existing bootstrap-runner entry point (Section 3.4) — this code
reuses the same per-record and per-run aggregation formulas as the
standard bootstrap procedure without altering either the replay
component or the bootstrap-runner component that Stage 2 uses directly.

**Stage 4 — Bootstrap.** For both Stage 2 and Stage 3, one bootstrap
run consists of drawing a resample of the same size as the source
dataset, with replacement, using a seeded random-number generator
shared across the whole campaign; a freshly constructed policy instance
is then replayed against that resample and the resulting matched steps
are aggregated into one run-level result. Thirty such runs are
performed per configuration, and the thirty run-level results are
further aggregated (Section 3.6) into the summary statistics reported
in Section 4.4. A single deterministic pass over the complete,
non-resampled dataset is additionally performed for every configuration
in Stages 2 and 3, providing the per-step series consumed by Stage 6.

**Stage 5 — Ablation.** The Random Critic, Reduced Context, and
Quality-only Reward configurations (Section 4.3) are each evaluated
against their respective reference baseline using the same bootstrap
procedure as Stage 4, through the framework's ablation component
(Section 3.6). The candidate arm of each ablation is additionally
replayed independently, under the same bootstrap procedure, to obtain
its own summary statistics and confidence interval in the same form as
every other configuration in this study.

**Stage 6 — Learning analysis.** For every configuration in Section
4.3, the single deterministic replay pass produced in Stage 4 (using
standard replay for every configuration except Sequential Learning
LinUCB, and sequential replay learning for that configuration) is
processed by the learning analysis component (Section 3.7) to compute
Reward per Step, Cumulative Reward, Moving Average Reward,
Instantaneous Regret, Cumulative Regret, Convergence Step, and a
learning-rate estimate for that configuration.

**Stage 7 — Statistics.** Pairwise statistical comparisons (Section
3.6) are computed between every candidate configuration and its
designated reference baseline, using each configuration's thirty
per-run average-reward values as the paired samples: Cold-Start and
Sequential Learning LinUCB (at every $\alpha$ value) against the
Heuristic Policy baseline; Sequential Learning LinUCB against Cold-Start
LinUCB at each matched $\alpha$ value; and each ablation candidate
against its reference baseline. Following this stage, the dataset
snapshot retained in Stage 1 is compared against the dataset's state at
this point to confirm no experiment stage modified the source data.

**Output generation.** Following Stage 7, four summary tables (Table 2:
overall policy comparison; Table 3: ablation study; Table 4: sequential
learning analysis; Table 5: learning analysis summary) and nine figures
(Figures 3-11: Reward Curve, Cumulative Reward, Moving Average Reward,
Instantaneous Regret, Cumulative Regret, Critic Selection Distribution,
Latency Comparison, Reward Distribution, and Convergence Analysis) are
generated from the results of Stages 2-7, together with machine-readable
exports of the same data.

## 4.6 Threats to Experimental Validity

The following limitations pertain to the design of the experimental
setup itself.

- **Synthetic data.** As stated in Section 4.2, no naturally occurring
  experience log was available; every experiment in this study was
  conducted against a single synthetically generated dataset. The
  generator's archetype weights, quality/latency/iteration
  distributions, and failure-rate probability were fixed at
  implementation time and were not calibrated against any external
  reference.
- **Single dataset seed.** All experiments share one dataset generation
  seed. No experiment in this study varies the dataset seed to assess
  sensitivity of the reported metrics to the specific realization of
  the synthetic log.
- **Fixed bootstrap sample count.** Thirty bootstrap runs were used for
  every configuration. This count was fixed for the whole campaign and
  was not selected via a power analysis or varied to assess its effect
  on confidence-interval width.
- **Single-pass learning-curve metrics.** Reward per Step, Cumulative
  Reward, Moving Average Reward, Cumulative Regret, and Convergence
  Step are each computed from one deterministic replay pass per
  configuration, not averaged across the thirty bootstrap runs used for
  the other metrics in Section 4.4; these two families of metrics
  therefore have different statistical bases within this experimental
  setup.
- **Match-rate dependency.** Every metric in Section 4.4 is computed
  only from replay steps at which a policy's selection matched the
  historically recorded selection (Section 3.4); the number of such
  steps varies by configuration and directly determines the sample size
  underlying that configuration's per-run and per-step statistics.
- **Fixed candidate action set.** All experiments use the same
  four-critic candidate action set; no experiment varies the number or
  composition of candidate actions.
- **Single exploration schedule.** Every LinUCB configuration in this
  study uses a constant exploration coefficient $\alpha$ for the
  duration of a run; no decaying, adaptive, or otherwise
  time-varying exploration schedule was evaluated.
- **Uncorrected multiple comparisons.** Section 4.5 (Stage 7) performs
  multiple pairwise statistical comparisons without applying a
  correction for multiple hypothesis testing; the significance
  threshold used throughout is the framework's default of 0.05,
  uncorrected.

## Chapter Summary

This section specified the five research questions this study is
designed to answer, the software environment and offline, synthetic-data
operating assumptions under which every experiment was conducted, the
six policy configurations evaluated, the twelve metrics recorded for
each, the seven-stage procedure by which the experiments were executed
— dataset generation, replay, sequential replay, bootstrap resampling,
ablation, learning-curve analysis, and statistical comparison — and the
limitations inherent to this experimental design. The results produced
by this procedure are reported in the following section.
