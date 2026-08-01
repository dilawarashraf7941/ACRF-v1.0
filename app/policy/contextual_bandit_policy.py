"""`ContextualBanditPolicy`: a stub `BasePolicy` reserving this policy's
place in `PolicyRegistry` for future implementation.

No contextual bandit algorithm (LinUCB, Thompson Sampling, or otherwise),
no reinforcement learning, no exploration, no reward updates, and no
learning of any kind are implemented here. `select_action` always raises
`NotImplementedError` with a clear message; this class exists purely so
`PolicyRegistry.register`/`.get`/`.list` (see `app/policy/registry.py`)
and `policy_engine_node` (see `app/graph/nodes.py`) already work,
unmodified, once a real implementation is written.
"""

from app.context import ContextVector
from app.policy.base import BasePolicy
from app.policy.models import PolicyDecision


class ContextualBanditPolicy(BasePolicy):
    """Stub for a future contextual bandit policy (e.g. LinUCB, Thompson Sampling).

    `select_action` is intentionally unimplemented — calling it always
    raises `NotImplementedError`. No scoring, exploration, or learning
    logic exists in this class.
    """

    policy_name = "ContextualBanditPolicy"
    policy_version = "0.0.0"

    def select_action(
        self, context: ContextVector, candidate_critics: list[str]
    ) -> PolicyDecision:
        """Always raise `NotImplementedError`; no contextual bandit algorithm exists yet.

        Args:
            context: Unused — accepted only to satisfy `BasePolicy`'s signature.
            candidate_critics: Unused — accepted only to satisfy `BasePolicy`'s signature.

        Raises:
            NotImplementedError: Always. This policy has no scoring,
                exploration, or learning logic implemented.
        """
        raise NotImplementedError(
            "ContextualBanditPolicy.select_action is not implemented. "
            "This class is a stub reserving the 'ContextualBanditPolicy' "
            "slot in PolicyRegistry for a future contextual bandit "
            "algorithm (e.g. LinUCB, Thompson Sampling). No bandit "
            "algorithm, exploration strategy, or learning is implemented "
            "in ACRF yet."
        )
