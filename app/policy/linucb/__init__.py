"""The LinUCB Contextual Bandit core.

A reusable, mathematically correct implementation of the disjoint
LinUCB algorithm (Li et al., 2010): `LinUCBArm` maintains one arm's
ridge-regression statistics (`A`, `A_inv`, `b`); `LinUCBPolicy` owns one
arm per action and selects/updates them from `ContextVector` inputs
only — never `AgentState`.

No Thompson Sampling, no replay buffer, no reinforcement learning, no
neural networks, and no exploration strategy other than LinUCB's own
confidence bound. This module is standalone: it is not imported by
`app/graph/nodes.py`, `policy_engine_node`, `router_node`,
`HeuristicPolicy`, or `PolicyRegistry`, and does not implement
`app.policy.base.BasePolicy`. Wiring it into the graph is future work.
"""

from app.policy.linucb.arm import LinUCBArm, context_feature_vector
from app.policy.linucb.models import LinUCBPrediction, LinUCBSelection
from app.policy.linucb.policy import LinUCBPolicy

__all__ = [
    "LinUCBArm",
    "LinUCBPolicy",
    "LinUCBPrediction",
    "LinUCBSelection",
    "context_feature_vector",
]
