# Abstract

Multi-agent Large Language Model (LLM) systems increasingly rely on
critic-based reasoning, in which specialized critic components
evaluate a generated response to support self-correction, and adaptive
routing has been proposed as a means of selecting which critic to
invoke based on context rather than a fixed rule. However, critic
selection in existing systems is typically governed by static,
predetermined rules that do not adapt to variation across tasks or
improve from observed outcomes. This paper presents the Adaptive
Critic Routing Framework (ACRF), developed to introduce a
context-aware, policy-guided critic-scoring layer alongside an
existing routing mechanism and to evaluate such a layer rigorously
before any live deployment. ACRF implements a deterministic heuristic
policy and a LinUCB contextual-bandit policy behind a common
interface, together with an explicitly separate, opt-in sequential
replay learning mode and an offline replay evaluation pipeline. This
pipeline was exercised through bootstrap-based statistical comparison,
ablation studies, and learning-curve analysis. Under the evaluated
offline protocol, cold-start LinUCB performed comparably to the
deterministic heuristic policy, and sequential replay learning did not
improve routing performance relative to either baseline; the
evaluation framework identified specific conditions under which the
adaptive policies studied here provided no measurable advantage over
deterministic routing. These findings are reported without
exaggeration or claims of performance superiority. The contribution of
this work lies in a reproducible, statistically grounded framework and
methodology for evaluating adaptive critic routing policies offline,
providing a foundation for future investigation of adaptive routing in
multi-agent LLM systems rather than a demonstrated performance
improvement.

**Word count:** 247

## Keywords

Adaptive Critic Routing; Multi-Agent LLM Systems; Contextual Bandits;
LinUCB; Offline Policy Evaluation; Sequential Replay Learning;
Self-Correction; Large Language Models
