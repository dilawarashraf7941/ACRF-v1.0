"""`LinUCBArm`: one arm's ridge-regression statistics for LinUCB.

Implements the standard (disjoint, per-arm) LinUCB algorithm of Li et al.
(2010), "A Contextual-Bandit Approach to Personalized News Article
Recommendation": each arm maintains its own design matrix `A`, its
inverse `A_inv`, and response vector `b`, updated online from
`(context, reward)` observations. No exploration strategy other than
LinUCB's own confidence bound is implemented — no Thompson Sampling, no
replay buffer, no neural network.
"""

import numpy as np

from app.context import ContextVector
from app.policy.linucb.models import LinUCBPrediction


def context_feature_vector(context: ContextVector) -> np.ndarray:
    """Build the numpy feature vector `x` LinUCB uses, from a `ContextVector`.

    Reads `context.features` in the order given by `context.feature_order`
    — never plain dict iteration order — so an identical `features`
    mapping always produces an identical vector, regardless of insertion
    order. Only `ContextVector` is accepted; `AgentState` is never read
    here or anywhere in this module.

    Args:
        context: The context to vectorize.

    Returns:
        A 1-D `float64` array of length `len(context.feature_order)`.

    Raises:
        ValueError: If `context.feature_order` is empty — LinUCB requires
            at least one feature to form a non-trivial context vector.
    """
    if not context.feature_order:
        raise ValueError(
            "ContextVector.feature_order is empty; LinUCB requires at least one feature."
        )
    return np.array(
        [context.features[name] for name in context.feature_order], dtype=np.float64
    )


class LinUCBArm:
    """One LinUCB arm: maintains `A`, `A_inv`, and `b` for a single action.

    Ridge-regression state, updated online:

    - `A` (`d x d`): `regularization * I` initially, then
      `A <- A + x xᵀ` on every `update`.
    - `A_inv` (`d x d`): `A`'s inverse, maintained incrementally via the
      Sherman-Morrison formula rather than repeated explicit inversion
      (see `update`).
    - `b` (`d`,): `0` initially, then `b <- b + r x` on every `update`.

    `predict` computes `theta = A_inv @ b` and the upper confidence bound
    `p = thetaᵀx + alpha * sqrt(xᵀ A_inv x)`.
    """

    def __init__(
        self,
        arm_id: str,
        dimension: int,
        alpha: float = 1.0,
        regularization: float = 1.0,
    ) -> None:
        """Create an arm with fresh (unobserved) statistics.

        Args:
            arm_id: Identifier of the action (critic) this arm represents.
            dimension: The context feature vector's length `d`. Fixed for
                the lifetime of this arm.
            alpha: The exploration coefficient (`alpha >= 0`). Larger
                values widen the confidence bound and favor exploration.
                A fixed constant chosen at construction time — never
                learned or updated.
            regularization: The ridge regularization strength
                (`regularization > 0`) used to initialize `A = regularization * I`.
                The standard LinUCB paper uses `1.0`.

        Raises:
            ValueError: If `dimension < 1`, `alpha < 0`, or
                `regularization <= 0`.
        """
        if dimension < 1:
            raise ValueError(f"dimension must be >= 1, got {dimension}")
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if regularization <= 0:
            raise ValueError(f"regularization must be > 0, got {regularization}")

        self.arm_id = arm_id
        self.dimension = dimension
        self.alpha = alpha
        self.regularization = regularization

        self.A: np.ndarray = regularization * np.eye(dimension, dtype=np.float64)
        self.A_inv: np.ndarray = np.eye(dimension, dtype=np.float64) / regularization
        self.b: np.ndarray = np.zeros(dimension, dtype=np.float64)

    def _validated_vector(self, context: ContextVector) -> np.ndarray:
        x = context_feature_vector(context)
        if x.shape != (self.dimension,):
            raise ValueError(
                f"context produced a {x.shape[0]}-dimensional vector; "
                f"arm {self.arm_id!r} expects dimension {self.dimension}"
            )
        return x

    def predict(self, context: ContextVector) -> LinUCBPrediction:
        """Compute this arm's LinUCB prediction for `context`.

        `theta = A_inv @ b`; `expected_reward = thetaᵀx`;
        `confidence_bonus = alpha * sqrt(xᵀ A_inv x)`.

        Args:
            context: The `ContextVector` to score. Never an `AgentState`.

        Returns:
            The resulting `LinUCBPrediction`.

        Raises:
            ValueError: If `context`'s feature vector's dimension does
                not match this arm's `dimension`.
        """
        x = self._validated_vector(context)
        theta = self.A_inv @ self.b
        expected_reward = float(theta @ x)
        variance = float(x @ self.A_inv @ x)
        confidence_bonus = float(self.alpha * np.sqrt(max(0.0, variance)))
        return LinUCBPrediction(
            arm_id=self.arm_id,
            expected_reward=expected_reward,
            confidence_bonus=confidence_bonus,
            upper_confidence_bound=expected_reward + confidence_bonus,
            context_id=context.context_id,
        )

    def update(self, context: ContextVector, reward: float) -> None:
        """Incorporate one `(context, reward)` observation.

        `A <- A + x xᵀ` and `b <- b + r x`. `A_inv` is updated via the
        Sherman-Morrison formula rather than recomputed from scratch:

            A_inv <- A_inv - (A_inv x)(A_inv x)ᵀ / (1 + xᵀ A_inv x)

        This is the exact inverse of the rank-1-updated `A` (Sherman-
        Morrison identity), and is numerically safe by construction: `A`
        starts positive definite (`regularization * I`) and a rank-1
        update `x xᵀ` (`x xᵀ` is positive semi-definite) preserves
        positive definiteness, so `A_inv` stays positive definite and
        `1 + xᵀ A_inv x >= 1` always — the denominator can never be zero
        or negative. `A_inv` is re-symmetrized (`(A_inv + A_invᵀ) / 2`)
        after every update to cancel floating-point asymmetry that would
        otherwise accumulate over many updates.

        Args:
            context: The `ContextVector` the reward was observed for.
            reward: The observed scalar reward.

        Raises:
            ValueError: If `context`'s feature vector's dimension does
                not match this arm's `dimension`.
        """
        x = self._validated_vector(context)

        A_inv_x = self.A_inv @ x
        denominator = 1.0 + float(x @ A_inv_x)
        self.A_inv = self.A_inv - np.outer(A_inv_x, A_inv_x) / denominator
        self.A_inv = (self.A_inv + self.A_inv.T) / 2.0

        self.A = self.A + np.outer(x, x)
        self.b = self.b + reward * x
