"""The Policy Abstraction Layer: a pluggable `BasePolicy` interface for critic selection.

Defines a common contract (`BasePolicy.select_action`) that every policy
implementation — the deterministic `HeuristicPolicy`, the stub
`ContextualBanditPolicy`, and any future `OfflineRLPolicy`/
`OnlineRLPolicy` — satisfies identically, plus a `PolicyRegistry` for
looking up "the current policy" by name without hard-coding a concrete
class at the call site.

No contextual bandit algorithm, no reinforcement learning, no PPO, no
Thompson Sampling, no LinUCB, no neural network, no policy learning, and
no reward/experience updates are implemented anywhere in this module.
"""

from app.policy.base import BasePolicy
from app.policy.contextual_bandit_policy import ContextualBanditPolicy
from app.policy.heuristic_policy import HeuristicPolicy
from app.policy.models import PolicyDecision
from app.policy.registry import DEFAULT_POLICY_REGISTRY, PolicyRegistry

__all__ = [
    "DEFAULT_POLICY_REGISTRY",
    "BasePolicy",
    "ContextualBanditPolicy",
    "HeuristicPolicy",
    "PolicyDecision",
    "PolicyRegistry",
]
