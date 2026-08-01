# 6. Discussion

This section interprets the results reported in Section 5 in light of
the architecture described in Section 3 and the experimental design
described in Section 4. No new experiments, measurements, or numerical
results are introduced; every claim in this section is derived from
evidence already presented.

## 6.1 Summary of Findings

Three principal observations emerge from Section 5. First, the
Heuristic Policy baseline and all four Cold-Start LinUCB configurations
recorded identical Average Reward, Average Quality, Average Latency,
Average Iterations, Match Rate, and confidence intervals (Table 2);
every paired comparison between a Cold-Start LinUCB configuration and
the Heuristic Policy baseline was reported as not statistically
significant, with a Cohen's $d$ of 0.0000 in each case (Table 6).
Second, the canonical Sequential Learning LinUCB configuration
($\alpha=1.0$) recorded a lower Average Reward, lower Average Quality,
an Average Latency essentially unchanged from every other configuration
in Table 2, higher Average Iterations, and a substantially lower Match
Rate — but this difference was **not** statistically significant
against either the Heuristic Policy baseline or the matched-$\alpha$
Cold-Start LinUCB configuration (Table 6). Across the evaluated
$\alpha$ sweep, Sequential Learning LinUCB's behavior separates into two
regimes rather than varying smoothly with $\alpha$: at $\alpha=0.25$ and
$\alpha=0.5$, every reported metric — matched-step count, Convergence
Step, and Final Cumulative Regret — is numerically identical to
Cold-Start LinUCB's; at $\alpha=1.0$ and $\alpha=2.0$, matched-step
count collapses from 123 to 4 and Convergence Step falls to 3 (Table 4).
Only the $\alpha=2.0$ comparison against Cold-Start LinUCB reached
statistical significance (Table 6). Third, of the three ablation
comparisons (Table 3), Random Critic and Quality-only Reward were each
reported as statistically significant departures from the Full ACRF
baseline, while Reduced Context recorded no measurable difference on
any reported metric.

## 6.2 Interpretation of Findings

### Heuristic Policy vs. Cold-Start LinUCB

The Heuristic Policy scores each candidate critic from nine extracted
state features (Section 3.3-A), and the LinUCB Policy scores each
candidate from a context vector using the same feature ordering
(Section 3.3-B); both order candidates by score and break ties
alphabetically by critic name. Section 3.2 states that the context
vector used throughout offline replay is built by a function
"deliberately kept independent of the live [Context Encoder]" and that
this offline builder derives its four features directly from
`ExperienceRecord` fields rather than from the nine features the
Heuristic Policy's weight table is defined over. Under this offline
context representation, the Heuristic Policy's weighted sum (equation
2) is evaluated over inputs it was not designed to score against
task-relevant content, so its score for every candidate reduces to the
same fixed base term regardless of the record being replayed, and
selection is decided entirely by the alphabetical tie-break. A freshly
constructed LinUCB arm begins with identical statistics for every
candidate ($A_a = \lambda I$, $b_a = \mathbf{0}$; equations 3-4), so its
predicted value is likewise identical across candidates for any input
context, and selection is again decided by the same alphabetical
tie-break, independent of $\alpha$. Both mechanisms therefore select
the same one candidate — alphabetically first among the four built-in
critic identifiers — for every replayed record, which is consistent
with the identical Match Rate (0.4017) recorded for the Heuristic
Policy and every Cold-Start LinUCB configuration in Table 2, and with
that value's proximity to the 40% share of records assigned to the
single-critic archetype described in Section 4.2.1.

### Sequential Replay Learning

Sequential Learning LinUCB differs from Cold-Start LinUCB only in that
its per-arm statistics are updated, via equations (3)-(4), after every
matched replay step within a single pass (Section 3.4). Once a
sufficient number of below-average-reward observations accumulate
against the initially tied candidate, that candidate's estimated value
falls below the still-untouched candidates' scores (which remain at
their initial, mutually identical level), and the policy's live
selection changes — *if* it changes at all. Table 4 and Table 5 show
that whether it changes depends sharply on $\alpha$ rather than varying
smoothly with it: at $\alpha=0.25$ and $\alpha=0.5$, every reported
metric for Sequential Learning LinUCB — 123 matched steps, a
Convergence Step of 119, a Final Cumulative Regret of 31.4848 — is
numerically identical to Cold-Start LinUCB's, meaning no selection
change occurred anywhere in either pass; at $\alpha=1.0$ and
$\alpha=2.0$, matched steps collapse to 4 and Convergence Step falls to
3, indicating an early selection change in both cases. This is
consistent with equation (3): the exploration term is scaled by
$\alpha$ while the point-estimate term is unaffected by it, so a given
downward movement in the initially selected candidate's estimated value
only overtakes the still-untouched candidates' scores once $\alpha$'s
contribution to their exploration bonus is large enough — a condition
this dataset's context representation (Section 3.2), which produces
only eleven distinct context vectors across the 300 stored records
(Section 4.2.1), satisfies at $\alpha=1.0$ and $\alpha=2.0$ but not at
$\alpha=0.25$ or $\alpha=0.5$.

### Match-Gated Replay

Section 3.4 states that only replay steps at which a policy's live
selection matches the historically recorded selection ever contribute
a reward or a training signal, in both replay modes. Once Sequential
Learning LinUCB's selection changes away from the initially tied
candidate, this same rule means the abandoned candidate's arm no longer
receives updates for the remainder of that pass, since it is no longer
selected. This mechanism only applies where a selection change actually
occurs: at $\alpha=0.25$ and $\alpha=0.5$, no change occurs (Section
6.2, above), and Sequential Learning LinUCB's Match Rate and
matched-step count are identical to Cold-Start LinUCB's. At
$\alpha=1.0$ (canonical) and $\alpha=2.0$, where a change does occur,
the substantially lower Match Rate (0.0072 at $\alpha=1.0$) and lower
matched-step count recorded for Sequential Learning LinUCB relative to
Cold-Start LinUCB (Table 2, Table 5) are consistent with this
mechanism: after a selection change, only the newly selected
candidate's share of the dataset — smaller than the 40% archetype
share the initial candidate held, for the archetype distribution
described in Section 4.2.1 — remains available to match.

### Exploration Parameter ($\alpha$)

Equation (3) scales the exploration term by $\alpha$ while the point
estimate term is unaffected by $\alpha$ at initialization (it is zero
for every untouched arm). A larger $\alpha$ therefore widens the
confidence bound of every still-untouched candidate relative to a given
downward movement in the initially selected candidate's estimated
value — but Table 4 shows this manifests as a threshold rather than a
gradient over the evaluated sweep: Convergence Step is 119 at both
$\alpha=0.25$ and $\alpha=0.5$ (identical to Cold-Start LinUCB, i.e. no
selection change occurs) and falls abruptly to 3 at both $\alpha=1.0$
and $\alpha=2.0$. Sequential Learning's Average Reward does not fall
monotonically with $\alpha$ either: it is 0.7749 at $\alpha=0.25$, rises
to 0.7875 at $\alpha=0.5$, then falls to 0.7653 at $\alpha=1.0$ and
0.5896 at $\alpha=2.0$ — consistent with reward being governed by
*whether* a selection change occurred within this specific 300-record
pass (Section 6.2) rather than by a smooth function of $\alpha$ itself.
Under Cold-Start LinUCB, by contrast, every candidate's exploration term
is scaled by the same $\alpha$ at the same initial context-independent
state (Section 3.3-B), so no value of $\alpha$ changes which candidate
the alphabetical tie-break selects; this is consistent with the
identical values recorded across all four Cold-Start $\alpha$ values in
Table 2.

### Random Critic

The Random Critic configuration selects independently of context
(Section 3.6), so its live selection matches the historically recorded
critic only when its uniformly random draw coincides with whatever was
recorded for that record. This is consistent with its lower Match Rate
(0.2280) relative to the Full ACRF baseline's 0.4017 (Table 3): the
baseline deterministically selects the archetype holding the largest
single-critic share of the dataset (Section 4.2.1) on every record,
while Random Critic's expected match probability is governed by the
relative sizes of all five archetypes in the dataset. The corresponding
reductions in Average Quality, and increases in Average Latency and
Average Iterations (Table 3), are consistent with the different mix of
archetypes contributing to Random Critic's matched steps relative to
the single, highest-quality/lowest-latency archetype (Section 4.2.1)
that dominates the Full ACRF baseline's matched steps.

### Reduced Context

The Reduced Context configuration wraps the same $\alpha=1.0$ LinUCB
Policy evaluated under Cold-Start LinUCB, masking a fraction of the
context vector's named features before scoring (Section 3.6). Because
an untrained LinUCB arm's predicted value does not depend on the
content of the context vector at all — both its point estimate and its
exploration term reduce to the same value for every candidate at
initialization, as discussed above — masking part of an input that does
not yet differentiate candidates cannot change which candidate is
selected. This is consistent with Reduced Context recording exactly
zero difference from the Full ACRF baseline on every metric in Table 3.

### Quality-only Reward

The Quality-only Reward configuration replaces the default weighted
reward strategy (equation 6) with a strategy that reports the record's
quality score alone (Section 3.6), while using the same $\alpha=1.0$
LinUCB Policy and therefore the same selection behavior as the Full
ACRF baseline. Table 3 reports a zero difference in Average Quality,
Average Latency, and Average Iterations between this configuration and
its baseline, consistent with unchanged selection behavior, alongside a
nonzero Average Reward difference of $-0.0094$. Under the default
weighted reward (equation 6), a completed record contributes a fixed
positive completion bonus not present in the quality-only definition;
given the 92% completion probability specified for the synthetic
dataset (Section 4.2.1), this is consistent with the weighted reward
strategy reporting a higher mean reward than the quality-only strategy
over the same underlying matched steps.

## 6.3 Comparison with Expectations

RQ1 asked whether policy-guided adaptive critic routing can outperform
non-adaptive routing strategies. The evidence reported in Section 5
does not support an affirmative answer for this study: Cold-Start
LinUCB was statistically indistinguishable from the Heuristic Policy
baseline at every tested $\alpha$ (Table 6), and the canonical
Sequential Learning LinUCB configuration ($\alpha=1.0$) was, likewise,
statistically indistinguishable from the Heuristic Policy baseline
(Table 6), despite recording a lower point-estimate Average Reward
(0.7653 vs. 0.7951). That point estimate is, in this revalidated
campaign, higher than the Random Critic configuration's point-estimate
Average Reward (0.7653 vs. 0.6916, Table 3) — a reversal of the
relative ordering reported previously — although no formal paired
comparison between these two specific configurations was performed in
this study. The proposed adaptive routing mechanism did not outperform
the deterministic heuristic in this experimental campaign, and this
outcome is reported directly rather than qualified: as discussed in
Section 6.2, both the Heuristic Policy and Cold-Start LinUCB reduce, in
this evaluation setting, to the same alphabetical tie-break, which
gives neither an opportunity to exhibit distinct — let alone
superior — behavior in the first place; the canonical Sequential
Learning configuration, though capable of departing from that
tie-break in principle (Section 6.2), did not do so significantly at
$\alpha=1.0$ in this specific, seed-determined pass.

RQ2 asked how sequential replay learning affects policy adaptation
relative to a cold-start instance of the same policy. The evidence
supports a more qualified answer than a uniform effect across the
$\alpha$ sweep would suggest: at $\alpha=0.25$ and $\alpha=0.5$,
sequential learning produced **no measurable change** in matched-step
count, Convergence Step, Final Cumulative Regret, or the statistical
comparison against Cold-Start LinUCB (Table 4, Table 6) — the trained
policy's behavior was indistinguishable from the untrained one
throughout both passes. At $\alpha=1.0$ (canonical) and $\alpha=2.0$,
sequential learning did materially change matched-step count and
Convergence Step (Table 4), but this change reached statistical
significance only at $\alpha=2.0$ (Table 6); the canonical $\alpha=1.0$
comparison against both Cold-Start LinUCB and the Heuristic Policy
baseline was not statistically significant, despite the large drop in
matched-step count (123 to 4). Where a change did occur and was
measurable, it was, as previously reported, in the direction of lower
reward relative to the untrained baseline (Table 4), not higher; no
configuration in this revalidated campaign showed sequential learning
improving on cold-start behavior.

RQ3 asked what effect the exploration coefficient $\alpha$ has on
routing behavior. The evidence supports different answers for the two
evaluation modes: $\alpha$ had no measurable effect on Cold-Start
LinUCB (Table 2). Its effect on Sequential Learning LinUCB was
substantial but threshold-like rather than monotonic (Table 4, Section
6.2): $\alpha=0.25$ and $\alpha=0.5$ produced no change in Convergence
Step or matched-step count relative to Cold-Start LinUCB, while
$\alpha=1.0$ and $\alpha=2.0$ produced an abrupt collapse in both;
Average Reward likewise did not vary monotonically across the sweep
(0.7749, 0.7875, 0.7653, 0.5896 at $\alpha=0.25$, $0.5$, $1.0$, $2.0$
respectively), rising slightly before falling rather than decreasing
throughout, consistent with reward being governed by whether a
selection change occurred rather than by a smooth function of $\alpha$
(Section 6.2).

RQ4 asked how sensitive ACRF's LinUCB policy is to a reduction in
context features. The evidence supports a null answer under the tested
condition: no measurable sensitivity was recorded (Table 3).

RQ5 asked how robust the measured policy value is to the definition of
the reward function. The evidence supports a qualified answer: routing
behavior itself was unaffected by the reward definition, but the
measured value was not — a statistically significant difference in
Average Reward was recorded between the two reward definitions applied
to the same matched steps (Table 3, Table 6).

## 6.4 Practical Implications

The results reported in Section 5, interpreted in Section 6.2, bear on
three practical questions.

**When ACRF-style adaptive routing should be used.** The adaptive
routing mechanism evaluated in this study produced a measurable
difference from cold-start behavior only at the higher exploration-
coefficient values tested, $\alpha=1.0$ and $\alpha=2.0$ (Sequential
Learning LinUCB), and even then only in the direction of lower measured
reward, reaching statistical significance only at $\alpha=2.0$ (Section
6.3). At $\alpha=0.25$ and $\alpha=0.5$, sequential learning produced no
measurable difference from cold-start behavior at all in this
revalidated campaign. On this evidence, deploying the LinUCB Policy
without either a pre-trained starting state or a lower, more
conservative exploration coefficient would not be expected to
outperform a deterministic routing rule, and may, at higher $\alpha$
values, depart from it in an uncontrolled direction.

**When adaptive routing is appropriate.** The Reduced Context and
Cold-Start results (Sections 6.2, 6.3) indicate that this framework's
adaptive mechanism cannot select differently from a deterministic
tie-break while every candidate arm remains untrained, regardless of
how differentiated the context representation is: a freshly constructed
LinUCB arm's score does not depend on the content of the context vector
it is queried with, only on the fact that every other untrained
candidate is queried with the same one (Section 6.2). Sequential
Learning LinUCB shows that once training moves one arm's estimate away
from this shared initial state, the context representation introduced
in Section 3.2 is differentiated enough to let candidates be scored
differently — but whether that difference actually produces a selection
change within a given replay pass depends sharply on the exploration
coefficient relative to how coarse that differentiation is (Section
6.2, Section 6.3), not on training alone.

**When deterministic routing is preferable.** Because Sequential
Learning LinUCB's Match Rate and Average Reward fell substantially and
irrecoverably (within a single pass) once its selection changed
(Section 6.2), deterministic routing is the lower-risk choice whenever
the cost of the specific irreversible mid-pass adaptation observed
here would be unacceptable, and no mechanism for detecting or reverting
such an adaptation is in place.

**Implications for future multi-agent LLM systems.** Two general
implications follow from the mechanisms identified in Section 6.2.
First, where a policy is scored from both a live, execution-time
context representation and a separately constructed offline or
training-time representation, feature parity between the two
representations determines whether learning observed offline is
representative of behavior in the live path; Section 3.2 documents that
this framework's two context builders are maintained independently, and
Section 6.2 identifies this as consistent with the Heuristic Policy's
observed context-invariant behavior under replay. Second, an online or
sequential update mechanism that can alter routing behavior based on a
small number of early observations, with no mechanism evaluated in this
study to revert or bound such a change, carries a demonstrated risk (in
this study's specific setting) of moving policy behavior away from a
reference configuration rather than toward a better one.

## 6.5 Limitations

**Synthetic dataset.** All results in Section 5 were obtained from one
synthetically generated dataset, whose archetypes fix a deterministic
association between critic identity and quality, latency, and iteration
distributions (Section 4.2.1). This association is what makes the
Random Critic and Sequential Learning results in Section 5
interpretable as directional effects at all; a dataset without such an
association, or with a different distribution of archetype shares,
would likely alter the specific Match Rate, Convergence Step, and
reward values reported, without necessarily altering the qualitative
mechanisms identified in Section 6.2.

**Offline replay.** Every metric in Section 5 is computed only from
replay steps at which a policy's live selection exactly matched the
historically recorded selection (Section 3.4). This requirement is
what produced the reduced sample underlying Sequential Learning LinUCB's
statistics once its selection changed (Section 6.2); an evaluation
methodology that did not require an exact match — for instance, one
using off-policy reward estimation instead — could report different
matched-step counts and reward statistics for the same underlying
policy behavior.

**Cold-start assumptions.** Every LinUCB configuration in this study
began from an untrained state (Section 4.3). The reported equivalence
between the four Cold-Start configurations, and the specific trajectory
of each Sequential Learning configuration, both follow from this
starting condition (Section 6.2); a policy initialized from prior
training would not necessarily reduce to the same alphabetical
tie-break at $t=0$, and the results reported here should not be read as
describing the behavior of a pre-trained instance of the same policy.

**Single-pass learning.** Sequential Learning LinUCB's reported
statistics reflect exactly one ordered pass over each bootstrap
resample of the dataset (Section 4.5). The specific Convergence Step
and Average Reward values in Table 4 are realizations of the ordering
produced by the campaign's shared random seed and are not necessarily
representative of behavior under repeated or continued exposure to the
same data, which this study did not evaluate.

**Constant alpha.** The exploration coefficient $\alpha$ was held fixed
for the duration of every run (Section 4.2). The relationship between
$\alpha$ and Convergence Step/Average Reward identified in Section 6.2
is a property of a non-adaptive exploration schedule; a schedule in
which $\alpha$ changes over the course of a pass was not evaluated and
could plausibly alter this relationship.

**Context representation.** As discussed in Section 6.2, the offline
context vector used throughout this study is deliberately independent
of the live Context Encoder and does not reproduce the nine features
the Heuristic Policy's weighting is defined over (Section 3.2). This
representational gap is consistent with, and is the most direct
available explanation for, the Heuristic Policy's and every Cold-Start
LinUCB configuration's context-invariant behavior reported in Section
5.1; a context representation that preserved these nine features under
replay could plausibly yield different Heuristic Policy behavior than
what was observed in this study. Separately, the offline context
producing this study's dataset takes only eleven distinct values across
the 300 stored records (Section 4.2.1); this coarseness, rather than
context invariance, is the most direct available explanation for why
Sequential Learning LinUCB's behavior separates sharply by $\alpha$
rather than varying smoothly (Section 6.2, Section 6.3) — a richer,
higher-cardinality context representation could plausibly narrow or
remove this threshold effect.

## 6.6 Future Work

The following directions are presented as possible extensions
consistent with the limitations identified in Section 6.5; none is
claimed to be necessary, and none was evaluated in this study.

- **Multiple replay passes.** Evaluating Sequential Learning LinUCB over
  more than one ordered pass over the same or additional data could
  indicate whether the reduced Match Rate observed after a single-pass
  selection change (Section 6.2) persists or is corrected given further
  exposure.
- **Real production logs.** Replacing the synthetic dataset (Section
  4.2.1) with recorded execution data from a deployed system would
  remove the synthetic-data limitation identified in Section 6.5, at
  the cost of requiring such a deployment to exist.
- **Alternative contextual bandits.** The policy interface described in
  Section 3.3 reserves a stub for a future contextual-bandit
  implementation beyond the disjoint LinUCB policy evaluated in this
  study; a non-disjoint, hybrid, or Thompson-sampling-based policy could
  be evaluated through the same offline evaluation pipeline (Section
  3.6) without requiring changes to it.
- **Policy gradient methods.** A policy class optimized via gradient-based
  methods, rather than the closed-form ridge-regression update used by
  LinUCB (Section 3.3-B), represents a further alternative not
  evaluated in this study.
- **Dynamic exploration.** An exploration coefficient that decays or
  otherwise varies over the course of a replay pass, rather than the
  constant $\alpha$ evaluated here, could plausibly reduce the
  sensitivity to $\alpha$ documented in Section 6.2 and Section 6.5.
- **Richer or aligned context features.** Extending the offline context
  builder to reproduce the nine features the Heuristic Policy's
  weighting depends on, or otherwise aligning it more closely with the
  live Context Encoder (Section 3.2), would directly address the
  representational gap identified as a limitation in Section 6.5. A
  higher-cardinality pre-decision context — beyond the eleven distinct
  values the current representation produces over this study's dataset
  (Section 6.5) — could also be investigated as a way to narrow the
  $\alpha$-threshold effect identified in Section 6.2 and Section 6.3.

## Discussion Summary

The results reported in Section 5 are consistent with a small number of
architectural mechanisms already described in Section 3: alphabetical
tie-breaking among simultaneously untrained candidates, which occurs
regardless of how differentiated the context representation is
(Heuristic Policy, Cold-Start LinUCB, Reduced Context); a match-gated
evaluation rule that, once training does move a policy's selection away
from that tie-break, withdraws further training signal from the
abandoned candidate within a single pass (Sequential Learning LinUCB, at
$\alpha=1.0$ and $\alpha=2.0$, but not at $\alpha=0.25$ or $\alpha=0.5$,
where no selection change occurred); and a reward-definition difference
applied to otherwise identical selection behavior (Quality-only Reward).
Measured against the study's five research questions, the evidence does
not support the hypothesis that the adaptive routing mechanisms
evaluated here outperform the deterministic heuristic baseline;
Cold-Start LinUCB matched it exactly at every tested $\alpha$, and the
canonical Sequential Learning LinUCB configuration ($\alpha=1.0$) was
statistically indistinguishable from it, with only the higher
$\alpha=2.0$ configuration performing significantly worse. This
outcome, and its consistency with the mechanisms identified above, is
reported without qualification. The limitations discussed in Section
6.5 — most notably the synthetic dataset, the untrained starting
condition applied to every LinUCB configuration, and the coarseness of
the offline context representation relative to the live Context
Encoder — bound the extent to which these findings generalize beyond
the specific experimental setting described in Section 4, and motivate
the extensions outlined in Section 6.6.
