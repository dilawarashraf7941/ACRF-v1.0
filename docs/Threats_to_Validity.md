# 7. Threats to Validity

This section assesses the validity of the experimental study described
in Sections 4-6 using the standard four-category taxonomy — internal,
external, construct, and conclusion validity. Threats already discussed
as limitations in Section 4.6 and Section 6.5 are not repeated here;
this section instead evaluates the methodological soundness of the
study design itself and, for each threat identified, states what
mitigation, if any, was applied.

## 7.1 Internal Validity

Internal validity concerns whether the differences observed between
configurations in Section 5 can be attributed to the factor being
varied (policy type, exploration coefficient, or ablation condition)
rather than to a confound in the experimental apparatus.

**Implementation assumptions.** The offline evaluation pipeline
operationalizes "selection" through a single, fixed decision rule —
exact set equality between a policy's live output and the historically
recorded critic identifiers, with any tie broken alphabetically
(Section 3.3, Section 3.4). This rule is an implementation choice, not
an empirical property of routing quality, and a different choice (for
example, partial-overlap matching) could change which experiences
contribute to a given configuration's statistics. This threat is
mitigated by applying the identical rule, unmodified, to every
configuration evaluated in this study (Section 4.3); any bias the rule
introduces is therefore common to all compared configurations rather
than favoring one over another.

**Synthetic dataset.** The synthetic generator fixes a deterministic
association between critic identity and the quality, latency, and
iteration values assigned to a record (Section 4.2.1) before any policy
is evaluated against it. Because this association is set independently
of, and prior to, policy evaluation, it constitutes a property of the
experimental apparatus rather than of any policy under test. This
threat is mitigated by evaluating every configuration in this study
against the same single dataset instance (Section 4.5), so the
association is held constant across all comparisons; it is not
mitigated with respect to the absolute values reported, which remain
specific to this one dataset.

**Replay procedure.** Because only matched replay steps contribute to
any reported metric (Section 3.4), different configurations can be
evaluated over different-sized, differently-composed effective samples
of the same underlying dataset — observed directly in this study as the
matched-step count falling from 123 for Cold-Start LinUCB to as low as
4 for Sequential Learning LinUCB at $\alpha \geq 1.0$ (Section 5.2).
This is an internal validity threat insofar as comparing two
configurations' aggregate statistics implicitly compares them over
different underlying samples. This threat is not fully mitigated within
this study; it is a structural consequence of the unbiased replay
method adopted in Section 3.4, whose alternative — imputing an outcome
for an unmatched experience — would introduce a more severe threat
(a fabricated counterfactual reward) that the chosen design was
specifically adopted to avoid.

**Learning protocol.** Sequential Learning LinUCB's per-run outcome
depends on the order in which matched records are presented within a
pass (Section 3.4), and that order is determined by a single seeded
bootstrap resample per run (Section 4.5). A different realization of
the same resampling procedure could produce a materially different
sequence of updates and therefore a different reported outcome for that
run. This threat is mitigated by aggregating thirty independent
bootstrap runs per configuration (Section 4.5); the reported statistics
and confidence intervals (Table 2, Table 4) reflect the resulting
distribution across these thirty realizations rather than any single
one.

## 7.2 External Validity

External validity concerns whether the findings in Section 5 and
Section 6 generalize beyond the specific conditions of this study.

**Generalization to real LLM systems.** Every result in this study was
obtained from an entirely offline evaluation over experience records
whose outcome values were synthetically generated rather than produced
by live critic evaluation of language-model-generated content (Section
3.1 documents the built-in critics as placeholder evaluators). Whether
the mechanisms identified in Section 6.2 hold when critics perform
genuine evaluation of real generated content is not established by this
study. This threat is partially mitigated by the offline evaluation
pipeline's design: it is explicitly agnostic to how `ExperienceRecord`s
are produced (Section 4.2), so the same replay, statistical-comparison,
and learning-analysis methodology (Sections 3.4, 3.6, 3.7) used
throughout this study could be applied to a log produced by a live
deployment without modification to the methodology itself; only the
data source would change.

**Generalization to different tasks.** All experiments in this study
used the same fixed candidate action set of four built-in critics and
the same task-routing rule (Section 3.1-F, Section 4.2). Whether the
findings extend to a system with a different or larger critic taxonomy,
or a different routing rule, is not addressed by this study. No
mitigation specific to this threat was applied; it is identified here
as an open question for future evaluation.

**Generalization to different datasets.** This study evaluated every
configuration against one synthetically generated dataset instance
(Section 4.2.1). This threat is partially mitigated by the fact that
the dataset generation procedure is fully parameterized and seeded
(Section 4.2.1) rather than hand-constructed on a per-experiment basis,
so alternative datasets could, in principle, be substituted through the
same generation interface without altering the evaluation methodology;
no such alternative dataset was evaluated in this study, so this
mitigation addresses reproducibility and extensibility rather than
generalization itself.

**Generalization to different policies.** Only the Heuristic Policy and
the disjoint LinUCB policy, together with ablation variants derived from
the latter, were evaluated (Section 4.3). This threat is mitigated at
the level of methodology, though not at the level of evidence: the
policy interface evaluated against (Section 3.3) and the evaluation
pipeline that consumes it (Section 3.6) are both defined independently
of any specific policy implementation, so the same experimental
procedure described in Section 4.5 is applicable to any other policy
conforming to that interface without modification; no other policy
class was evaluated in this study.

## 7.3 Construct Validity

Construct validity concerns whether the metrics defined in Section 4.4
adequately operationalize "routing quality," the property this study
is intended to assess.

**Reward.** Reward is constructed as a fixed weighted combination of a
quality term, a completion bonus, and cost/latency/correction penalties
(Section 3.5). Its validity as a measure of routing quality depends
entirely on these weights corresponding to a deployment's actual
priorities. This study provides direct evidence that the construct is
sensitive to this weighting: the Quality-only Reward ablation applied a
different, single-term reward definition to selection behavior that
Table 3 reports as otherwise identical to its baseline, and recorded a
statistically significant difference in mean reward (Section 5.4,
Section 5.5). Reward, as operationalized in this study, therefore
reflects a specific fixed weighting choice rather than a single,
unambiguous notion of routing quality. No mitigation for this
construct-level dependency was applied beyond reporting results under
both reward definitions as a distinct ablation condition.

**Quality.** The `aggregated_quality_score` construct is intended to
represent a critic-assessed measure of output correctness. In this
study, this value is generated synthetically per the archetype
distributions in Section 4.2.1 rather than produced by the framework's
own (placeholder) critic evaluation logic (Section 3.1). Quality, as
measured here, is therefore a construct standing in for critic-assessed
correctness, not an instance of it; this substitution is inherent to
the offline, synthetic-data study design (Section 4.2) and was not
otherwise mitigated.

**Latency.** Latency measures execution time and is a reasonable
construct for routing *efficiency*, but it does not, by itself,
indicate whether a routing decision was correct: a policy could route
to a fast critic that is nonetheless a poor match for the task. This
study mitigates the risk of over-interpreting latency in isolation by
reporting it jointly with reward, quality, iterations, and match rate
for every configuration (Section 5.1), rather than as a standalone
indicator of routing quality.

**Iterations.** The recorded iteration count reflects the correction
policy's downstream decision to apply an additional correction cycle
(Section 3.1-H) at least as directly as it reflects the critic-routing
decision that this study's policies are responsible for. Its
relationship to routing quality specifically, as distinct from
correction-policy behavior, is therefore indirect. This was not
separately mitigated; iterations is reported alongside the other
metrics in Section 4.4 rather than in isolation, for the same reason
given for latency above.

**Match rate.** Match rate measures agreement between a policy's live
selection and a stored historical selection (Section 3.4), which is
itself the output of whatever process originally produced the
experience log — in this study, the synthetic generator's archetype
assignment (Section 4.2.1), not an independently validated notion of
correct routing. A policy could, in principle, route differently from
the historical record while making an equal or better decision, and
this study's design cannot distinguish that case from a genuinely worse
policy, since only matched experiences contribute to any other
reported metric (Section 3.4). This is not mitigated within the present
design; it is an accepted property of the replay method adopted in
Section 3.4, whose purpose is to guarantee unbiasedness of the metrics
it does report, not to independently validate the reference selections
matching is defined against.

Taken together, no single metric in Section 4.4 is treated as a
sufficient measure of routing quality on its own; the joint reporting
of all twelve metrics for every configuration (Section 5) is the
principal mitigation applied against over-relying on any one
construct, and the interpretations in Section 6.2 draw on multiple
metrics in combination rather than any single one.

## 7.4 Conclusion Validity

Conclusion validity concerns whether the statistical procedures applied
in Section 3.6 and used to produce Section 5's results support the
conclusions drawn from them in Section 6.

**Statistical power.** Every reported test used a fixed sample size of
thirty bootstrap runs per configuration (Section 4.5), not a size
derived from a formal power analysis (Section 4.6). This is a threat to
the reliability of any *non-significant* result specifically — a true
but small effect could fail to reach significance at this sample size.
This threat is directly relevant to the canonical Sequential Learning
LinUCB configuration ($\alpha=1.0$): despite a substantial point-estimate
reward difference and a matched-step count that collapsed from 123 to 4
(Section 5.2), its comparison against both the Heuristic Policy baseline
and Cold-Start LinUCB was not statistically significant ($p=0.5561$,
Section 5.4), consistent with the small effective sample size a
collapsed matched-step count produces. It is a lesser threat to the
three *significant* results reported in Table 6, each of which was
reported at $p \leq 0.0016$ (Section 5.4), well below what a marginal,
underpowered detection would typically produce. No formal power
analysis was retrospectively applied to any of the non-significant
results in this study.

**Bootstrap.** The thirty runs underlying each configuration's
statistics are resamples, with replacement, of one fixed 300-record
dataset (Section 4.5), rather than thirty independently collected
samples. This is mitigated by using an evaluation methodology explicitly
designed for resampled data throughout: the percentile-based confidence
interval for a single configuration's own reward distribution and the
automatically-selected paired hypothesis test for two configurations'
paired per-run values (Section 3.6) are both defined over exactly this
kind of resampled, paired data, rather than applying formulas that
assume fully independent samples.

**Paired tests.** The statistical comparison procedure selects between
a paired $t$-test and a Wilcoxon signed-rank test via a Shapiro-Wilk
normality check, and additionally defines explicit, separate handling
for a single paired observation and for the case in which every paired
difference is identical (Section 3.6). This explicit handling is itself
a mitigation: rather than passing degenerate inputs — which occurred
routinely in this study, given the Cold-Start LinUCB and Reduced
Context results reported in Section 5 — through to a general-purpose
test implementation with unspecified behavior on such inputs, the
procedure resolves them analytically and reports a determinate,
reproducible outcome (`degenerate_zero_variance` in Table 6).

**Effect sizes.** Cohen's $d$ for paired samples (equation 11) was
reported alongside every $p$-value in Table 6, mitigating the risk of
treating statistical significance as equivalent to practical magnitude.
This measure's own validity is inherited from the reward construct
discussed in Section 7.3: $d$ quantifies a standardized difference in
mean reward specifically, and carries whatever construct-validity
limitations apply to reward as a measure of routing quality.

**Confidence intervals.** Two different interval constructions were
used in this study for two different quantities — a percentile interval
for a single configuration's own reward distribution across bootstrap
runs, and a $t$-distribution interval for the mean difference between
two paired configurations (Section 3.6). Conflating these would
threaten the validity of any conclusion drawn from a reported interval.
This is mitigated by consistent, distinct labeling throughout Section 5
(”95% CI (Reward)” in Table 2 versus “95% CI (Difference)” in Table 6).

**Multiple comparisons.** Twelve pairwise statistical tests were
performed in this study (Table 6) without applying a correction for
multiple hypothesis testing (Section 4.6), at an uncorrected
significance threshold of 0.05. Applying a Bonferroni correction for
twelve comparisons yields an adjusted threshold of approximately 0.0042.
Checked against this adjusted threshold post hoc, every one of the
three comparisons reported as significant in Table 6 remains significant
($p \leq 0.0016$ in every case), while every comparison reported as not
significant in Table 6 remains not significant ($p$ ranging from
$0.3173$ to $1.0000$ across the nine non-significant comparisons). The
significance outcomes reported in Section 5 and interpreted in Section
6 are therefore unchanged under this more conservative correction,
which serves as the mitigation applied to this threat within the
present study.

## Summary

This section evaluated the study described in Sections 4-6 against the
four standard validity categories. Internal validity is threatened
primarily by the replay method's dependence on matched-step composition
and by the order-dependence of sequential learning outcomes, mitigated
respectively by applying the matching rule uniformly across
configurations and by aggregating thirty bootstrap runs per
configuration. External validity is threatened by the study's reliance
on offline, synthetic evaluation and a narrow scope of tasks, datasets,
and policy classes, mitigated only at the level of methodology
(an evaluation pipeline and policy interface both designed to be
data-source- and policy-agnostic) rather than at the level of evidence.
Construct validity is threatened by each individual metric's partial
and indirect relationship to "routing quality," mitigated by reporting
all twelve metrics jointly rather than relying on any single one.
Conclusion validity is threatened by a fixed, non-power-analyzed sample
size and by twelve uncorrected pairwise comparisons; the former is
addressed only for the significant results reported, and the latter was
checked post hoc against a Bonferroni-adjusted threshold, under which
every significance outcome reported in Section 5 was unchanged. None of
these threats was found, on the evidence available within this study,
to invalidate the specific findings reported in Section 5 and
interpreted in Section 6; several nonetheless bound the scope within
which those findings should be read, consistent with the limitations
identified in Section 4.6 and Section 6.5.
