"""`BasePolicy`: the abstract interface every ACRF policy implements.

Defines only the contract — no contextual bandit algorithm, no
reinforcement learning, no PPO, no Thompson Sampling, no LinUCB, no
neural network, no policy learning, and no reward/experience updates are
implemented here. Every concrete policy (`HeuristicPolicy`,
`ContextualBanditPolicy`, and any future `OfflineRLPolicy`/
`OnlineRLPolicy`) implements `select_action` against this same signature,
so `policy_engine_node` (see `app/graph/nodes.py`) never needs to change
regardless of which concrete policy is active.
"""

from abc import ABC, abstractmethod

from app.context import ContextVector
from app.policy.models import PolicyDecision


class BasePolicy(ABC):
    """Abstract interface for a policy that selects critics from a `ContextVector`.

    Concrete subclasses must set `policy_name`/`policy_version` and
    implement `select_action`. No implementation is provided here.
    """

    policy_name: str = "BasePolicy"
    policy_version: str = "0.0.0"

    @abstractmethod
    def select_action(
        self, context: ContextVector, candidate_critics: list[str]
    ) -> PolicyDecision:
        """Select critics from `candidate_critics` given `context`.

        Args:
            context: The encoded, numeric observation of the current
                state (see `app/context`).
            candidate_critics: The critic identifiers eligible for selection.

        Returns:
            The resulting `PolicyDecision`.
        """
        raise NotImplementedError
