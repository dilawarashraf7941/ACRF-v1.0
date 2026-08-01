# LinUCB Contextual Bandit Core

A reusable, mathematically correct implementation of **disjoint LinUCB**
(Li et al., 2010, "A Contextual-Bandit Approach to Personalized News
Article Recommendation"). This module implements the algorithm's core
only — it is deliberately **not wired into anything else in ACRF**.

> **Scope:** LinUCB only. No Thompson Sampling, no replay buffer, no
> reinforcement learning, no neural networks, and no exploration
> strategy other than LinUCB's own upper confidence bound. This module
> is not imported by `app/graph/nodes.py`; `policy_engine_node` and
> `router_node` are unmodified; `HeuristicPolicy` is not replaced; this
> module does not implement `app.policy.base.BasePolicy` and is not
> registered in `PolicyRegistry`. Graph topology is untouched.

## Files

| File | Responsibility |
|---|---|
| `models.py` | `LinUCBPrediction`, `LinUCBSelection` — the structured, immutable outputs. |
| `arm.py` | `LinUCBArm` — one arm's `A`/`A_inv`/`b` statistics, plus `context_feature_vector`. |
| `policy.py` | `LinUCBPolicy` — owns one arm per action, exposes `select_action`/`update`. |

## Context, never `AgentState`

`context_feature_vector(context: ContextVector) -> np.ndarray` is the
only bridge between ACRF state and this module. It reads
`context.features` in the order given by `context.feature_order` (not
dict iteration order), so identical `features` always produce an
identical vector. `LinUCBArm` and `LinUCBPolicy` never see, import, or
depend on `AgentState`.

## Mathematical formulation

Each arm `a` (one per action/critic) maintains ridge-regression
statistics over observed `(x, r)` pairs, `x` the context feature vector
and `r` the observed reward:

- `A_a` (`d x d`): initialized to `regularization * I` (regularization
  default `1.0`, matching the original paper).
- `b_a` (`d`,): initialized to `0`.
- `A_a⁻¹` maintained alongside `A_a`.

**Prediction**, for a context `x`:

```
theta_a = A_a⁻¹ b_a
p_a(x)  = theta_aᵀ x + alpha * sqrt(xᵀ A_a⁻¹ x)
```

`theta_aᵀ x` is the point estimate (`expected_reward`); `alpha *
sqrt(xᵀ A_a⁻¹ x)` is the exploration bonus (`confidence_bonus`); their
sum is the upper confidence bound (`upper_confidence_bound`) arms are
ranked by. `alpha >= 0` is a fixed constant set at construction time —
never learned, never updated.

**Update**, given an observed `(x, r)` for arm `a`:

```
A_a <- A_a + x xᵀ
b_a <- b_a + r x
```

## `A_inv` numerical stability

`A_a⁻¹` is maintained incrementally via the **Sherman-Morrison
formula** rather than recomputed by explicit matrix inversion on every
update:

```
A_inv <- A_inv - (A_inv x)(A_inv x)ᵀ / (1 + xᵀ A_inv x)
```

This is the exact inverse of the rank-1-updated `A_a + x xᵀ`. It is
numerically safe by construction, not by a defensive check:

- `A_a` starts positive definite (`regularization * I`, regularization
  `> 0`).
- Adding `x xᵀ` (positive semi-definite) to a positive definite matrix
  keeps it positive definite, for any `x` and any number of updates.
- A positive definite `A_a⁻¹` implies `xᵀ A_inv x >= 0` for any `x`, so
  the denominator `1 + xᵀ A_inv x >= 1` — **it can never be zero or
  negative**, so this division never fails.

`A_inv` is re-symmetrized (`(A_inv + A_invᵀ) / 2`) after every update to
cancel the tiny floating-point asymmetry that would otherwise compound
over many updates. `confidence_bonus`'s `xᵀ A_inv x` is additionally
clamped to `>= 0` before the square root, guarding against a
floating-point value that should be `0` landing at `-1e-16`.

## `LinUCBArm`

```python
arm = LinUCBArm(arm_id="LogicCritic", dimension=27, alpha=1.0, regularization=1.0)
prediction = arm.predict(context)   # -> LinUCBPrediction
arm.update(context, reward=0.8)     # A, A_inv, b updated in place
```

`dimension` is fixed for the arm's lifetime; `predict`/`update` raise
`ValueError` if a `context`'s feature vector length doesn't match it.

## `LinUCBPolicy`

```python
policy = LinUCBPolicy(alpha=1.0, regularization=1.0)
selection = policy.select_action(context, ["LogicCritic", "CodeCritic", "FactCritic", "MetaCritic"])
# selection.selected_action, selection.predictions[...]
policy.update(context, action=selection.selected_action, reward=0.8)
```

Arms are created lazily, on first use of an action name — unobserved
arms predict `expected_reward=0.0` (a "no signal yet" prior) and a
confidence bonus that depends only on `x` and `alpha`, matching the
literature's cold-start behavior. Context dimension `d` is fixed to the
first `ContextVector` this policy ever scores; every subsequent call is
checked against it.

Selection ranks candidates by `upper_confidence_bound`, ties broken by
action name in ascending alphabetical order — the same convention
`CriticRanking` uses (see `app/policy_engine/ranking.py`) — so selection
is deterministic given identical arm state and context.

## Explicit non-goals

- No Thompson Sampling, epsilon-greedy, or any exploration strategy
  other than LinUCB's own upper confidence bound.
- No replay buffer.
- No reinforcement learning, PPO, Q-learning, or neural networks.
- No integration into `app/graph/nodes.py`, `policy_engine_node`, or
  `router_node` — all unmodified.
- No replacement of `HeuristicPolicy`, no registration in
  `PolicyRegistry`, no implementation of `app.policy.base.BasePolicy`.
- No change to graph topology.
