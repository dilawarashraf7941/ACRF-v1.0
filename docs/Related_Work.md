# 2. Related Work

This section reviews prior work relevant to the Adaptive Critic Routing
Framework (ACRF) presented in this paper: multi-agent LLM systems,
critic-based self-correction, contextual bandit algorithms, offline
policy evaluation, and adaptive routing approaches in LLM systems.
Section 2.6 identifies the gap in this literature that motivates the
framework described in Section 3.

## 2.1 Multi-Agent LLM Systems

Multi-agent LLM architectures decompose a task across multiple
specialized model invocations or roles, rather than relying on a single
monolithic generation call, and have been applied to improve
performance on tasks requiring multi-step reasoning, tool use, or
extended context management [2]. A widely used instance of this
pattern is the planner-worker architecture, in which a planning
component decomposes a task into a structured sequence of sub-steps or
sub-goals, and one or more worker components subsequently execute those
sub-steps [6]. Beyond this two-role pattern, broader agent-collaboration
frameworks coordinate several agents through mechanisms such as
message passing, debate, or voting, dividing responsibility for a task
across agents that may hold different roles, tools, or perspectives
[7], [8]. Task decomposition — breaking a complex task into smaller,
individually tractable sub-tasks — is a recurring element across these
architectures, whether performed explicitly by a dedicated planning
component or emerging from the interaction of multiple agents [6].

These architectures have demonstrated improved performance on tasks
that a single-pass generation call handles poorly, and their modularity
allows individual components to be specialized, reused, or replaced
independently. Their principal limitations are architectural complexity
and the additional latency and computational cost introduced by
multiple model invocations. Of particular relevance to this work, the
logic governing how components are coordinated — including which
downstream component handles a given sub-task or output — is, in the
reviewed literature, typically implemented as a fixed, hand-designed
mechanism rather than as a component that is itself learned, evaluated,
or adapted based on observed outcomes [2].

## 2.2 Critic-Based Self-Correction

A closely related line of work addresses how a generated output is
evaluated and revised. Self-reflection techniques have a generative
model evaluate or critique its own prior output and use that critique
to inform a subsequent revision, without necessarily invoking a
separate model instance for the evaluation step [3]. Critic agents
generalize this idea by introducing a dedicated evaluation component —
potentially a distinct model instance or a rule-based evaluator — that
assesses a candidate output against defined criteria and returns
structured feedback to the generating component [4]. Iterative
refinement applies generation and critique repeatedly, across multiple
rounds, until a stopping criterion such as a quality threshold or an
iteration budget is reached [9]. Multi-critic evaluation extends this
further by employing several critics, each specialized for a different
evaluation dimension — for example, logical consistency, factual
accuracy, or code correctness — and combining their individual
judgments into an aggregate assessment [4].

Across this literature, which critic or critics are invoked for a given
output is generally determined in one of two ways: either every
available critic is invoked unconditionally, incurring a computational
cost proportional to the number of critics regardless of their
individual relevance to the task at hand, or a fixed, hand-specified
rule maps a coarse task category to a predetermined critic or subset of
critics [2]. In either case, the decision of *which* critic to invoke
— as distinct from how a critic's feedback is subsequently used — is
treated as a fixed architectural property of the system rather than as
an object of adaptation. To the extent that critic selection is
revisited at all in this literature, it is typically as part of a
broader system redesign rather than through a policy that adapts this
specific decision based on context and observed outcomes; adaptive,
context-sensitive routing of critic invocation has received
comparatively little dedicated attention as an independent decision
problem [2].

## 2.3 Contextual Bandits

The multi-armed bandit problem formalizes repeated decision-making
under uncertainty: an agent repeatedly selects among a fixed set of
actions, observes a reward for the selected action alone, and seeks to
maximize cumulative reward despite initially unknown action payoffs
[10]. The contextual bandit extends this formulation by additionally
conditioning each decision on a feature vector — the context —
describing the current situation, allowing a policy to select different
actions for different contexts rather than converging on a single,
globally best action [5]. LinUCB is a widely used contextual bandit
algorithm that models each action's expected reward as a linear
function of the context and selects actions according to an
upper-confidence-bound rule that balances the estimated expected reward
against the policy's uncertainty about that estimate [11]. More
broadly, online adaptation in this setting refers to a policy updating
its internal statistics incrementally as new context-action-reward
observations become available, allowing its behavior to change over
time without requiring a separate, offline training phase before any
observations are used [5].

Critic routing is naturally expressible within this formulation: a
discrete set of candidate critics corresponds to the bandit's action
set, a structured representation of the current task or execution state
corresponds to the context, and an observed measure of output quality
following a critic's involvement corresponds to the reward. This
correspondence makes established contextual bandit algorithms, such as
LinUCB, a natural candidate mechanism for adaptive critic routing, in
place of a routing algorithm designed specifically for this purpose.

## 2.4 Offline Policy Evaluation

A separate body of work addresses how a candidate decision policy can
be evaluated without deploying it live. Logged data evaluation assesses
a candidate policy using previously recorded interaction data rather
than through further interaction with a live system, avoiding the cost
and risk associated with online experimentation [12]. Within this area,
replay methods for off-policy evaluation of logged bandit feedback
restrict evaluation to logged instances at which the candidate policy's
action agrees with the action that was actually logged, yielding an
evaluation that does not require estimating the outcome of an action
that was never observed, at the cost of discarding logged instances at
which the two disagree [12]. Bootstrap evaluation techniques estimate
the sampling distribution, confidence intervals, and statistical
significance of an evaluation metric through resampling from a single
fixed dataset, and are widely used when only one realization of logged
data is available for evaluation [13]. Offline reinforcement learning
constitutes a broader family of methods that learn or improve a policy
entirely from a fixed, previously collected dataset, without further
interaction with the environment during training [14].

These techniques share the advantage of enabling policy comparison, and
in the case of offline reinforcement learning, policy improvement,
without the cost, risk, or latency of live experimentation, and of
supporting reproducible evaluation from a fixed, shared dataset. Their
principal limitations are a dependence on the coverage and quality of
the available logged data, a reduction in effective sample size for
unbiased techniques such as the replay method, which discard logged
instances that do not match the candidate policy's decision, and, for
offline reinforcement learning specifically, sensitivity to
distributional mismatch between the logged data and the policy being
evaluated or trained [12], [14].

## 2.5 Adaptive Routing in LLM Systems

Several routing strategies have been applied within LLM-based systems
to direct an input toward one of several downstream components. Static
routing applies a fixed mapping from a coarse input category to a
single downstream component, uniformly and without regard to
finer-grained variation within that category [15]. Rule-based routing
generalizes this with a small set of hand-authored conditional rules,
typically based on simple heuristics rather than a learned scoring
function [15]. Confidence-based routing instead conditions the routing
decision on a model's own confidence or uncertainty estimate for a
given input, for example escalating to a more capable or specialized
component when confidence is low [16]. Mixture-of-experts routing
learns a gating function that assigns each input to one or more
specialized sub-networks, or experts, with the gating function typically
trained end-to-end alongside the experts themselves as part of a single
differentiable model [17].

Each of these approaches differs from the policy-guided adaptive
critic routing investigated in this work. Static and rule-based routing
do not adapt to observed outcomes at all. Confidence-based routing
adapts to an uncertainty signal but does not employ a context-dependent
policy that scores and selects among discrete downstream components in
the manner of a contextual bandit, nor is it typically accompanied by
an offline, statistically grounded evaluation methodology applied prior
to deployment. Mixture-of-experts routing is learned, but as an
integral, differentiable part of a single end-to-end model, jointly
optimized with the experts it routes to, rather than as a separately
evaluable decision policy operating over discrete, independently
existing critic components; nor is it typically evaluated through
offline replay against previously logged routing decisions, as
investigated in this work.

## 2.6 Research Gap

The literature reviewed above addresses the concerns motivating this
work largely in isolation. Multi-agent LLM and critic-based
self-correction systems (Sections 2.1, 2.2) generally treat critic
selection as a fixed architectural property, without an accompanying
mechanism for adapting that decision or a methodology for evaluating
candidate adaptations before deployment. Contextual bandit algorithms
(Section 2.3) provide a general, well-studied mechanism for
context-aware action selection, but their application specifically to
critic routing within a multi-agent LLM pipeline — introduced in a way
that keeps the study of such a policy separate from a pipeline's live
routing decision — is not established by the multi-agent and
self-correction literature reviewed in Sections 2.1 and 2.2. Offline
policy evaluation techniques, including replay-based methods, bootstrap
evaluation, and offline reinforcement learning (Section 2.4), are
studied largely independently of critic routing as an application
domain. Existing routing approaches applied within LLM systems (Section
2.5) are either non-adaptive or, where adaptive, trained as an integral
part of a differentiable end-to-end model rather than as a separately
trainable and offline-evaluable policy layer situated alongside an
existing routing mechanism.

Based on this review, existing work does not present an integrated
framework combining adaptive critic routing, context-aware policy
selection, sequential replay learning, offline evaluation, and
comprehensive statistical analysis within a single, evaluable system.
The Adaptive Critic Routing Framework described in this paper addresses
this identified gap: it introduces a context-aware, policy-guided
critic-scoring layer situated alongside an existing routing mechanism
without altering it; supports both a purely evaluative offline replay
mode and an explicitly separate, opt-in sequential replay learning
mode; and evaluates both through a methodology combining
bootstrap-resampled statistical comparison, ablation analysis, and
learning-curve analysis, applied consistently within a single system.
This is presented as an integration of concerns that the reviewed
literature largely addresses separately, applied to the specific
problem of critic routing, and not as a claim of superiority over any
individual technique or system reviewed in this section — several of
which address problems, such as end-to-end differentiable routing or
large-scale offline reinforcement learning, that are outside the scope
of the present work.

The following section presents the methodology by which this
integration is realized, beginning with the overall architecture of the
Adaptive Critic Routing Framework.
