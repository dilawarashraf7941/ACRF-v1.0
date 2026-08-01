# 8. Conclusion

## 8.1 Research Summary

Multi-stage agent pipelines that generate a response, critique it
through one or more specialized critics, and conditionally correct it
typically decide *which* critic to invoke through a fixed,
non-adaptive rule. Whether a learned or heuristic policy layer can be
introduced to guide this decision — without destabilizing the pipeline
it sits alongside, and while remaining possible to study rigorously and
reproducibly before any such policy is trusted with live routing
decisions — is the problem this work addresses. The Adaptive Critic
Routing Framework (ACRF) v1.0 was designed and implemented as a
deterministic, LangGraph-based execution pipeline (Section 3.1) in
which a policy-scoring layer operates alongside, but does not override,
the pipeline's existing deterministic routing rule, together with an
entirely offline evaluation suite (Sections 3.4, 3.6, 3.7) capable of
replaying, comparing, and analyzing candidate routing policies against
previously recorded execution data without invoking the live pipeline.
This design was motivated by the need to evaluate adaptive routing
policies under statistically rigorous, reproducible conditions before
any such policy could be considered for a live routing role.

## 8.2 Main Contributions

This work makes the following contributions, each of which was both
implemented and experimentally exercised in this study.

1. **The Adaptive Critic Routing Framework architecture** (Section
   3.1): a deterministic, graph-structured execution pipeline in which
   planning, worker generation, error-feature extraction, critic
   execution, correction, and evaluation are each implemented as
   discrete, independently testable stages, with a policy-scoring layer
   architecturally decoupled from the pipeline's live routing decision.
2. **A pluggable, policy-guided critic-scoring interface** (Section
   3.3), comprising a deterministic heuristic scoring policy and a
   disjoint LinUCB contextual-bandit policy behind a common abstraction,
   evaluated under standard (non-learning) and sequential-learning
   replay modes and across a range of exploration-coefficient values
   (Section 5.1, 5.2).
3. **A context representation layer** (Section 3.2) comprising both a
   live, execution-time context encoder and an independent
   offline-replay context builder, whose feature composition and its
   consequences for replayed policy behavior were directly evaluated
   through a dedicated context-reduction ablation (Section 5.5).
4. **A sequential replay learning mode** (Section 3.4) that trains a
   policy incrementally from replayed historical data under an
   explicit, match-gated correctness constraint, added as an
   additional, opt-in capability alongside an unmodified, purely
   evaluative replay mode, and evaluated across four exploration-
   coefficient settings (Section 5.2, 5.3).
5. **A comprehensive, offline evaluation methodology** (Sections 3.6,
   3.7): bootstrap-resampled policy evaluation, an automated paired
   statistical comparison procedure with effect sizes and confidence
   intervals, a seven-configuration ablation framework, and a
   learning-curve analysis component reporting cumulative reward,
   regret, moving averages, and convergence behavior — all applied
   consistently across every configuration evaluated in this study
   (Section 4, Section 5).

Consistent with the findings reported in Section 5 and interpreted in
Section 6, these contributions constitute a working framework and
evaluation methodology for studying adaptive critic routing; they do
not, on their own, constitute evidence that the specific policies
evaluated in this study improve routing performance relative to a
deterministic baseline. That evidence is reported separately in
Section 8.3.

## 8.3 Key Findings

The following findings are drawn directly from Section 5 and Section 6,
without new analysis.

- Cold-start LinUCB performed identically to the deterministic
  heuristic baseline across every evaluated exploration-coefficient
  value; no statistically significant difference was observed between
  them (Section 5.1, Section 5.4).
- Sequential replay learning, under the offline, match-gated protocol
  evaluated in this study, did not improve routing performance relative
  to either the deterministic heuristic or the untrained (cold-start)
  policy at any tested exploration-coefficient value. It produced a
  statistically significant *decrease* in measured reward only at
  $\alpha=2.0$; at the canonical $\alpha=1.0$ it recorded a lower
  point-estimate reward that was not statistically significant, and at
  $\alpha=0.25$ and $\alpha=0.5$ it produced no measurable difference
  from cold-start behavior at all (Section 5.2, Section 5.4).
- Match-gated replay strongly influenced sequential learning behavior
  wherever a selection change actually occurred: once a policy's live
  selection changed within a replay pass, the fraction of historical
  data contributing further training signal or evaluation credit fell
  sharply. Whether such a change occurred at all depended on the
  exploration coefficient in a threshold-like, not gradual, way — no
  change occurred at $\alpha=0.25$ or $\alpha=0.5$, while an abrupt
  change occurred at $\alpha=1.0$ and $\alpha=2.0$ (Section 5.2, Section
  5.3).
- The exploration coefficient had no measurable effect on cold-start
  routing behavior. Its effect on sequential learning outcomes was
  substantial but discontinuous rather than monotonic: Convergence Step
  and matched-step count were unchanged from cold-start at
  $\alpha=0.25$/$0.5$ and collapsed abruptly at $\alpha=1.0$/$2.0$, and
  measured reward did not decrease monotonically with $\alpha$ across
  the sweep (Section 5.2).
- Uniformly random critic selection produced a statistically
  significant decrease in reward, quality, and latency/iteration
  efficiency relative to the deterministic baseline (Section 5.5).
- Reducing the context features available to the routing policy
  produced no measurable difference in any reported metric under the
  evaluated (untrained) condition (Section 5.5).
- Substituting an alternative, quality-only reward definition for the
  framework's default weighted reward produced a statistically
  significant difference in measured reward despite unchanged routing
  behavior (Section 5.5).

## 8.4 Research Implications

For **multi-agent LLM systems**, this work provides a concrete,
evaluated example of decoupling a candidate adaptive-routing policy
from a system's live routing decision so that the policy can be
studied offline before, or independently of, any decision to deploy it;
the results in Section 5 illustrate that such offline evaluation can
surface a policy's behavior — including unfavorable behavior — before
it affects a live system. This implication is scoped to systems whose
routing decisions can similarly be logged and replayed; it is not
presented as applicable to every multi-agent architecture.

For **adaptive routing** specifically, this work's findings (Section
8.3) indicate that introducing a contextual-bandit policy does not
automatically yield improved routing decisions. A cold-start policy
requires at least one arm's estimate to move away from a shared,
untrained initial state before its candidates can be scored
differently, regardless of how differentiated the context
representation already is; and once that condition is met, whether an
exploration-driven, sequential-learning mechanism actually departs from
the untrained baseline — and in which direction — depends sharply on
the exploration coefficient relative to the granularity of the context
representation (Section 6.2, Section 6.3, Section 6.5). Neither
condition reliably produced an improvement over the deterministic
baseline in this study's evaluated configurations. This is offered as
evidence relevant to adaptive-routing research generally, not as a
claim that adaptive routing cannot succeed under other conditions.

For **offline policy evaluation**, this work demonstrates a complete,
reproducible pipeline — replay, bootstrap resampling, automated paired
statistical testing, ablation analysis, and learning-curve analysis —
applied consistently to compare six routing configurations, including
one trained through an explicitly separate, opt-in sequential-learning
mode whose introduction did not alter the behavior of the pre-existing,
purely evaluative replay mode (Section 3.4, Section 4.5). This
methodology is presented as a reusable pattern for evaluating future
routing policies within this or similarly structured systems, not as a
general-purpose evaluation methodology independent of that context.

For **critic-based self-correction**, this work's scope was limited to
the routing decision that selects which critic evaluates a given
output; the critics' own evaluation logic and the correction policy
that acts on their output (Section 3.1) were held fixed throughout and
were not themselves subject to adaptive routing or learning in this
study. The implications above concern the routing layer specifically
and should not be read as claims about critic evaluation quality or
correction-policy behavior.

## 8.5 Future Research

The findings and limitations discussed in Section 6.5 and Section 7
motivate several directions for future work, presented as opportunities
rather than requirements.

- **Real production datasets.** Evaluating the same framework and
  methodology against recorded data from a live deployment, rather than
  the synthetically generated dataset used throughout this study, would
  address the external-validity limitations discussed in Section 6.5
  and Section 7.2.
- **Multi-pass replay learning.** Extending sequential learning beyond
  the single deterministic pass evaluated in this study could indicate
  whether an early, unfavorable adaptation of the kind observed here is
  corrected given continued exposure to the same or additional data.
- **Dynamic exploration schedules.** Replacing the constant exploration
  coefficient evaluated in this study with a schedule that varies over
  the course of a replay pass could be explored as a way to moderate
  the exploration-related sensitivity identified in Section 6.2.
- **Alternative contextual bandits.** The policy interface evaluated in
  this study reserves a stub for future contextual-bandit
  implementations beyond the disjoint LinUCB policy studied here
  (Section 3.3); non-disjoint, hybrid, or otherwise structured bandit
  formulations remain unevaluated.
- **Reinforcement learning methods.** Policy classes trained through
  reinforcement learning or policy-gradient methods, rather than the
  closed-form update used by the LinUCB policy evaluated here, were
  outside the scope of this study and represent a further direction.
- **Richer context representations.** Aligning the offline-replay
  context representation more closely with the live context encoder, or
  extending either to capture additional signal, could be investigated
  as a way to address the context-representation limitation identified
  in Section 6.5 and Section 7.3.

## Closing Remarks

This work implemented and evaluated a complete, reproducible
adaptive critic routing framework and offline evaluation methodology,
comprising a policy-guided routing layer, a sequential replay learning
mode, and a statistically rigorous evaluation pipeline covering
bootstrap comparison, ablation analysis, and learning-curve analysis.
Applied to the six routing configurations studied here, this
methodology did not find evidence that the evaluated adaptive-routing
policies outperform a deterministic heuristic baseline, and it found
that the evaluated sequential learning protocol reduced routing
performance, to a degree that depended sharply on the exploration
coefficient, at the higher end of the tested range. These findings are
reported as the outcome of the evaluation, not as a limitation of the
framework's capacity to produce them: the principal contribution of
this work is a framework and methodology capable of surfacing exactly
this kind of result — including a negative one — under reproducible,
statistically supported conditions, and of doing so without risk to a
live system. The significance of this work lies in that capability, and
in the specific, evidence-grounded understanding of when the evaluated
adaptive routing mechanisms did and did not provide value, rather than
in a claim that adaptive critic routing was shown to outperform simpler
alternatives in this study.
