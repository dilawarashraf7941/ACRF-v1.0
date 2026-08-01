# 1. Introduction

## 1.1 Background

Large Language Models (LLMs) have become the primary generative
component in an increasingly wide range of automated reasoning and
content-generation systems [1]. As these systems are asked to perform
more complex, multi-step tasks, single-model, single-pass generation
has proven insufficient on its own, motivating architectures in which
multiple specialized components collaborate to produce a final output
— commonly referred to as multi-agent LLM systems [2]. In such systems,
a generative component (often termed a *worker*) produces an initial
response, which is then subject to one or more downstream evaluation
steps before being accepted, revised, or escalated.

*Self-correction* refers to the capacity of such a system to detect and
remedy deficiencies in its own output, typically by feeding evaluative
feedback back into a further generation or revision step [3].
*Critic-based reasoning* is one mechanism by which this feedback is
produced: a dedicated critic component evaluates a candidate output
against some criterion — logical consistency, factual accuracy, code
correctness, or the quality of a prior correction attempt, among others
— and returns a structured judgment that downstream components consume
[4]. Because different critics are typically specialized for different
evaluation criteria, and because invoking every available critic on
every output is not always necessary or economical, the *routing*
decision of which critic (or critics) to invoke for a given output is
itself a significant design choice.

*Adaptive decision making* — selecting an action based on the
observed state of a system rather than through a fixed, predetermined
rule — has been studied extensively in sequential decision problems,
including contextual bandit formulations in which an action is chosen
based on a feature representation of the current context and refined
over time from observed outcomes [5]. Applying this idea to critic
routing raises the possibility that the choice of which critic to
invoke could itself be guided by the observed characteristics of a
task, rather than fixed in advance. As multi-agent LLM systems are
deployed in increasingly varied and resource-constrained settings,
adaptive critic routing — selecting critics based on context, rather
than through a single fixed rule applied uniformly to every task — has
correspondingly grown in relevance, both as a potential efficiency
mechanism and as an open question in its own right: whether, and under
what conditions, an adaptive routing policy provides any advantage over
a simpler, deterministic alternative.

## 1.2 Problem Statement

Despite this growing relevance, critic routing in existing multi-agent
LLM pipelines is typically implemented through static, fixed rules —
for example, mapping a coarse task category to a single predetermined
critic — applied uniformly regardless of the specific characteristics
of an individual task instance. Such fixed routing policies are simple
to implement and easy to reason about, but they do not adapt to
variation within a task category, and they provide no mechanism by
which routing behavior could improve as more execution data becomes
available. At the same time, invoking every available critic
unconditionally, in order to avoid this limitation, incurs a
computational and latency cost that scales with the number of critics
available, regardless of whether every critic is actually informative
for a given task. This tension — between the simplicity and
predictability of fixed routing and the potential efficiency of
adaptive, context-sensitive routing — motivates a central question this
work addresses: whether a policy-guided routing layer can be introduced
into a multi-agent critic pipeline in a way that is both architecturally
safe (that is, does not destabilize the pipeline's existing behavior)
and rigorously evaluable, and, if so, whether such a layer offers any
measurable advantage over the fixed routing rule it would sit alongside.

## 1.3 Research Motivation

Addressing the problem above motivates four specific design choices
carried through the remainder of this work. First, adaptive critic
routing is needed because a fixed routing rule cannot, by construction,
account for variation in task characteristics that a learned or
heuristic policy could in principle exploit; investigating whether such
a policy provides any advantage requires first constructing one and a
means of evaluating it fairly. Second, contextual decision making —
scoring routing candidates against a structured, numeric representation
of the current task state, rather than a coarse category label — is
useful because it is the mechanism by which any adaptive policy could
plausibly differentiate between task instances that a fixed rule treats
identically. Third, offline evaluation is important because introducing
an untested adaptive policy directly into a live pipeline risks
destabilizing behavior that a fixed rule currently provides reliably;
evaluating candidate policies against previously recorded execution
data, before any live deployment, allows their behavior to be studied
under controlled, repeatable conditions and compared against a
deterministic baseline without that risk. Fourth, sequential replay
learning was investigated because a policy that can improve from
observed outcomes is a natural extension of a purely evaluative
adaptive policy, and because offline replay data — the same data used
for evaluation — is a natural, low-risk substrate on which to
investigate whether such improvement occurs, provided any training
procedure applied to it preserves the same correctness guarantees as
the evaluation procedure itself.

## 1.4 Research Objectives

This work pursues the following objectives.

- Design an Adaptive Critic Routing Framework in which a policy-guided
  scoring layer can be introduced alongside an existing, fixed critic
  routing rule without altering that rule's behavior.
- Investigate policy-guided critic selection through a deterministic
  heuristic scoring policy and a contextual-bandit policy, evaluated
  under a common interface.
- Study contextual routing strategies by constructing a structured,
  numeric representation of task state suitable for scoring routing
  candidates, and by evaluating routing policies' sensitivity to the
  features available in that representation.
- Develop an offline replay evaluation methodology capable of
  comparing candidate routing policies against previously recorded
  execution data, including statistically grounded comparison,
  ablation analysis, and reward-definition sensitivity analysis.
- Analyze adaptive learning behavior by introducing a sequential,
  replay-based learning mode and characterizing the resulting policy
  behavior through reward, regret, and convergence analysis, without
  altering the framework's purely evaluative replay mode.

## 1.5 Research Contributions

This work contributes the following, each of which is both designed
and evaluated in the sections that follow.

1. **An Adaptive Critic Routing Framework architecture**: a
   deterministic, graph-structured execution pipeline in which
   planning, worker generation, error-feature extraction, critic
   execution, correction, and evaluation are implemented as discrete
   stages, with a policy-scoring layer architecturally decoupled from
   the pipeline's existing routing decision.
2. **A pluggable, policy-guided critic-scoring interface**, comprising
   a deterministic heuristic scoring policy and a disjoint LinUCB
   contextual-bandit policy behind a common abstraction.
3. **A context representation layer**, comprising both a live,
   execution-time context encoder and an independent offline-replay
   context builder.
4. **A sequential replay learning mode** that trains a policy
   incrementally from replayed historical data under an explicit,
   match-gated correctness constraint, introduced as an additional,
   opt-in capability alongside an unmodified, purely evaluative replay
   mode.
5. **A comprehensive, offline evaluation methodology**, comprising
   bootstrap-resampled policy evaluation, an automated paired
   statistical comparison procedure with effect sizes and confidence
   intervals, an ablation framework, and a learning-curve analysis
   component.

## 1.6 Paper Organization

The remainder of this paper is organized as follows. Section 2 reviews
related work on multi-agent LLM pipelines, critic-based self-correction,
contextual bandit algorithms, and offline policy evaluation, and
situates the present work relative to it. Section 3 presents the
methodology underlying the Adaptive Critic Routing Framework, including
its overall architecture, context representation, policy-guided
routing mechanisms, sequential replay learning mode, reward function,
evaluation framework, and learning analysis components. Section 4
describes the experimental setup used to evaluate this framework,
including the research questions investigated, the experimental
environment and dataset, the evaluated baseline and ablation
configurations, the evaluation metrics, and the experimental procedure.
Section 5 reports the results obtained from this procedure. Section 6
discusses and interprets these results, compares them against the
research questions posed in Section 4, and considers their practical
implications and limitations. Section 7 assesses the internal,
external, construct, and conclusion validity of the experimental study.
Section 8 concludes the paper, summarizing its contributions, key
findings, and directions for future work. The following section begins
this progression by reviewing the body of prior work this framework
builds upon.
