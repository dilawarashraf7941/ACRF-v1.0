"""`LinUCBPolicy`: the reusable LinUCB contextual bandit core.

Owns one `LinUCBArm` per action (critic), created lazily the first time
an action is seen. No exploration strategy other than LinUCB's own
confidence bound is implemented — no Thompson Sampling, no replay
buffer, no reinforcement learning, no neural network.

This class is intentionally **not** wired into `app/graph/nodes.py`,
`policy_engine_node`, `router_node`, `HeuristicPolicy`, or
`PolicyRegistry`. It does not implement `app.policy.base.BasePolicy` —
its method signatures (`select_action(context, actions)`,
`update(context, action, reward)`) are deliberately its own, standalone
contract, matching this task's spec rather than the existing
`BasePolicy` interface. Wiring it into the graph is future work.
"""

from app.context import ContextVector
from app.policy.linucb.arm import LinUCBArm, context_feature_vector
from app.policy.linucb.models import LinUCBPrediction, LinUCBSelection


class LinUCBPolicy:
    """Selects and updates LinUCB arms, one per action (critic).

    All arms share the same `alpha` and `regularization`, and the same
    context dimension `d` — fixed to the length of the first
    `ContextVector` this policy ever sees, and checked on every
    subsequent call.
    """

    def __init__(self, alpha: float = 1.0, regularization: float = 1.0) -> None:
        """Create a policy with no arms yet; arms are created lazily.

        Args:
            alpha: The exploration coefficient passed to every arm this
                policy creates (`alpha >= 0`).
            regularization: The ridge regularization strength passed to
                every arm this policy creates (`regularization > 0`).

        Raises:
            ValueError: If `alpha < 0` or `regularization <= 0`.
        """
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if regularization <= 0:
            raise ValueError(f"regularization must be > 0, got {regularization}")

        self.alpha = alpha
        self.regularization = regularization
        self._arms: dict[str, LinUCBArm] = {}
        self._dimension: int | None = None

    @property
    def arms(self) -> dict[str, LinUCBArm]:
        """A shallow copy of the action -> `LinUCBArm` mapping, for introspection."""
        return dict(self._arms)

    def _arm_for(self, action: str, context: ContextVector) -> LinUCBArm:
        x = context_feature_vector(context)
        if self._dimension is None:
            self._dimension = x.shape[0]
        elif x.shape[0] != self._dimension:
            raise ValueError(
                f"context produced a {x.shape[0]}-dimensional vector; "
                f"this policy was initialized with dimension {self._dimension}"
            )

        arm = self._arms.get(action)
        if arm is None:
            arm = LinUCBArm(
                arm_id=action,
                dimension=self._dimension,
                alpha=self.alpha,
                regularization=self.regularization,
            )
            self._arms[action] = arm
        return arm

    def select_action(self, context: ContextVector, actions: list[str]) -> LinUCBSelection:
        """Predict every candidate action's LinUCB score and select the best one.

        Any action not seen before gets a fresh `LinUCBArm` (unobserved
        prior: `expected_reward=0`, confidence bonus scaled by `x`
        alone). Selection ranks by `upper_confidence_bound`, ties broken
        by action name in ascending alphabetical order — the same
        deterministic tie-breaking convention `CriticRanking` uses (see
        `app/policy_engine/ranking.py`).

        Args:
            context: The `ContextVector` to score every action against.
                Never an `AgentState`.
            actions: The candidate action (critic) identifiers.

        Returns:
            A `LinUCBSelection` with one `LinUCBPrediction` per action and
            the selected action.

        Raises:
            ValueError: If `actions` is empty, or if `context`'s feature
                vector's dimension does not match this policy's fixed
                dimension.
        """
        if not actions:
            raise ValueError("actions must be non-empty")

        predictions: dict[str, LinUCBPrediction] = {
            action: self._arm_for(action, context).predict(context) for action in actions
        }
        ranked = sorted(
            actions, key=lambda name: (-predictions[name].upper_confidence_bound, name)
        )
        selected_action = ranked[0]

        return LinUCBSelection(
            selected_action=selected_action,
            predictions=predictions,
            alpha=self.alpha,
            context_id=context.context_id,
        )

    def update(self, context: ContextVector, action: str, reward: float) -> None:
        """Incorporate one `(context, reward)` observation for `action`.

        Args:
            context: The `ContextVector` the reward was observed for.
                Never an `AgentState`.
            action: The action (critic) identifier the reward is
                attributed to. Gets a fresh `LinUCBArm` if not seen
                before.
            reward: The observed scalar reward.

        Raises:
            ValueError: If `context`'s feature vector's dimension does
                not match this policy's fixed dimension.
        """
        arm = self._arm_for(action, context)
        arm.update(context, reward)
