"""Reward computation strategies: `BaseRewardStrategy` and `WeightedRewardStrategy`.

Converts a completed `ExperienceRecord` into a deterministic `RewardSignal`.
No reinforcement learning, no contextual bandits, no PPO/DQN/Q-learning,
no neural networks, no replay buffers, no policy optimization, and no
randomness — every weight below is a fixed constant chosen at
implementation time, and every component is a simple, bounded, pure
function of its input.
"""

from abc import ABC, abstractmethod

from app.experience import ExperienceRecord
from app.reward.models import RewardSignal

QUALITY_REWARD_WEIGHT = 1.0
"""Multiplier applied to a clamped [0, 1] `aggregated_quality_score` to produce `quality_reward`."""

CORRECTION_PENALTY_PER_ITERATION = 0.1
"""Penalty added per correction iteration."""

MAX_CORRECTION_PENALTY = 0.5
"""Upper bound on `correction_penalty`, regardless of how many iterations occurred."""

COST_PENALTY_SCALE = 0.01
"""Multiplier applied to `estimated_cost` to produce `cost_penalty`."""

MAX_COST_PENALTY = 0.5
"""Upper bound on `cost_penalty`, regardless of how large `estimated_cost` is."""

LATENCY_PENALTY_SCALE = 0.05
"""Multiplier applied to `latency` (in seconds) to produce `latency_penalty`."""

MAX_LATENCY_PENALTY = 0.5
"""Upper bound on `latency_penalty`, regardless of how large `latency` is."""

COMPLETION_BONUS = 0.2
"""Added to `completion_bonus` when `execution_status == "completed"`."""

FAILURE_PENALTY = 0.3
"""Subtracted from `completion_bonus` (making it negative) when `execution_status == "failed"`."""


def _compute_quality_reward(aggregated_quality_score: float | None) -> float:
    """Convert an aggregated quality score into a positive reward component.

    Degrades gracefully: a missing score contributes `0.0` rather than
    raising. A score outside `[0.0, 1.0]` (e.g. from a future critic using
    a different scale) is clamped rather than allowed to distort the
    total reward unboundedly.

    Args:
        aggregated_quality_score: `ExperienceRecord.aggregated_quality_score`.

    Returns:
        `clamp(aggregated_quality_score, 0.0, 1.0) * QUALITY_REWARD_WEIGHT`,
        or `0.0` if `aggregated_quality_score` is `None`.
    """
    if aggregated_quality_score is None:
        return 0.0
    return max(0.0, min(1.0, aggregated_quality_score)) * QUALITY_REWARD_WEIGHT


def _compute_correction_penalty(iterations: int) -> float:
    """Convert a correction-iteration count into a bounded penalty.

    Args:
        iterations: `ExperienceRecord.iterations`.

    Returns:
        `min(MAX_CORRECTION_PENALTY, iterations * CORRECTION_PENALTY_PER_ITERATION)`,
        or `0.0` if `iterations <= 0`.
    """
    if iterations <= 0:
        return 0.0
    return min(MAX_CORRECTION_PENALTY, iterations * CORRECTION_PENALTY_PER_ITERATION)


def _compute_cost_penalty(estimated_cost: float | None) -> float:
    """Convert an estimated cost into a bounded penalty.

    Degrades gracefully: a missing cost contributes `0.0` rather than
    raising, and a negative value (which should not occur, but is not
    validated here) is treated as `0.0` rather than becoming a bonus.

    Args:
        estimated_cost: `ExperienceRecord.estimated_cost`.

    Returns:
        `min(MAX_COST_PENALTY, max(0.0, estimated_cost) * COST_PENALTY_SCALE)`,
        or `0.0` if `estimated_cost` is `None`.
    """
    if estimated_cost is None:
        return 0.0
    return min(MAX_COST_PENALTY, max(0.0, estimated_cost) * COST_PENALTY_SCALE)


def _compute_latency_penalty(latency: float | None) -> float:
    """Convert a latency measurement into a bounded penalty.

    Degrades gracefully: a missing latency contributes `0.0` rather than
    raising, and a negative value is treated as `0.0`.

    Args:
        latency: `ExperienceRecord.latency`.

    Returns:
        `min(MAX_LATENCY_PENALTY, max(0.0, latency) * LATENCY_PENALTY_SCALE)`,
        or `0.0` if `latency` is `None`.
    """
    if latency is None:
        return 0.0
    return min(MAX_LATENCY_PENALTY, max(0.0, latency) * LATENCY_PENALTY_SCALE)


def _compute_completion_bonus(execution_status: str) -> float:
    """Convert the terminal execution status into a signed bonus/penalty.

    Any status other than `"completed"` or `"failed"` (e.g. `"pending"`,
    unknown future statuses) degrades gracefully to a neutral `0.0`.

    Args:
        execution_status: `ExperienceRecord.execution_status`.

    Returns:
        `COMPLETION_BONUS` if `"completed"`, `-FAILURE_PENALTY` if
        `"failed"`, else `0.0`.
    """
    if execution_status == "completed":
        return COMPLETION_BONUS
    if execution_status == "failed":
        return -FAILURE_PENALTY
    return 0.0


def _compute_confidence(
    aggregated_quality_score: float | None,
    estimated_cost: float | None,
    latency: float | None,
) -> float:
    """Measure how complete this reward computation's optional inputs were.

    Args:
        aggregated_quality_score: `ExperienceRecord.aggregated_quality_score`.
        estimated_cost: `ExperienceRecord.estimated_cost`.
        latency: `ExperienceRecord.latency`.

    Returns:
        The fraction (0.0 to 1.0) of the three optional signals above
        that were not `None`.
    """
    signals = (aggregated_quality_score, estimated_cost, latency)
    available = sum(1 for signal in signals if signal is not None)
    return available / len(signals)


class BaseRewardStrategy(ABC):
    """Abstract interface for a strategy that converts an `ExperienceRecord` into a `RewardSignal`.

    Defines only the contract; no reward algorithm is implemented here.
    Concrete subclasses must set `strategy_name` and implement `compute`.
    """

    strategy_name: str = "BaseRewardStrategy"

    @abstractmethod
    def compute(self, experience: ExperienceRecord) -> RewardSignal:
        """Compute a `RewardSignal` from `experience`.

        Args:
            experience: The completed execution experience to score.

        Returns:
            The resulting `RewardSignal`.
        """
        raise NotImplementedError


class WeightedRewardStrategy(BaseRewardStrategy):
    """Deterministic, fixed-weight reward strategy.

    Combines five independent, bounded components — `quality_reward`,
    `completion_bonus`, `cost_penalty`, `latency_penalty`, and
    `correction_penalty` — using fixed weights chosen at implementation
    time. No component is learned, fitted, or randomized.
    """

    strategy_name = "WeightedRewardStrategy"

    def compute(self, experience: ExperienceRecord) -> RewardSignal:
        """Compute a `RewardSignal` from `experience` via fixed weighted components.

        Args:
            experience: The completed execution experience to score.

        Returns:
            A `RewardSignal` with every component populated and
            `reward = quality_reward + completion_bonus - cost_penalty -
            latency_penalty - correction_penalty`.
        """
        quality_reward = _compute_quality_reward(experience.aggregated_quality_score)
        correction_penalty = _compute_correction_penalty(experience.iterations)
        cost_penalty = _compute_cost_penalty(experience.estimated_cost)
        latency_penalty = _compute_latency_penalty(experience.latency)
        completion_bonus = _compute_completion_bonus(experience.execution_status)
        efficiency_penalty = cost_penalty + latency_penalty
        confidence = _compute_confidence(
            experience.aggregated_quality_score, experience.estimated_cost, experience.latency
        )

        reward = (
            quality_reward + completion_bonus - cost_penalty - latency_penalty - correction_penalty
        )

        explanation = (
            f"reward={reward:.4f} = quality_reward={quality_reward:.4f} "
            f"+ completion_bonus={completion_bonus:.4f} "
            f"- cost_penalty={cost_penalty:.4f} - latency_penalty={latency_penalty:.4f} "
            f"- correction_penalty={correction_penalty:.4f}. "
            f"(efficiency_penalty={efficiency_penalty:.4f} is cost_penalty + "
            f"latency_penalty, reported for reference and not subtracted again.)"
        )

        return RewardSignal(
            reward=round(reward, 6),
            quality_reward=round(quality_reward, 6),
            efficiency_penalty=round(efficiency_penalty, 6),
            cost_penalty=round(cost_penalty, 6),
            latency_penalty=round(latency_penalty, 6),
            correction_penalty=round(correction_penalty, 6),
            completion_bonus=round(completion_bonus, 6),
            confidence=round(confidence, 6),
            strategy=self.strategy_name,
            explanation=explanation,
            metadata={
                "experience_id": experience.experience_id,
                "inputs": {
                    "aggregated_quality_score": experience.aggregated_quality_score,
                    "iterations": experience.iterations,
                    "estimated_cost": experience.estimated_cost,
                    "latency": experience.latency,
                    "execution_status": experience.execution_status,
                },
                "weights": {
                    "quality_reward_weight": QUALITY_REWARD_WEIGHT,
                    "correction_penalty_per_iteration": CORRECTION_PENALTY_PER_ITERATION,
                    "max_correction_penalty": MAX_CORRECTION_PENALTY,
                    "cost_penalty_scale": COST_PENALTY_SCALE,
                    "max_cost_penalty": MAX_COST_PENALTY,
                    "latency_penalty_scale": LATENCY_PENALTY_SCALE,
                    "max_latency_penalty": MAX_LATENCY_PENALTY,
                    "completion_bonus": COMPLETION_BONUS,
                    "failure_penalty": FAILURE_PENALTY,
                },
            },
        )
