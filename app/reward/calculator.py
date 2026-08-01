"""`RewardCalculator`: the sole entry point converting an `ExperienceRecord`
into a `RewardSignal`.

Responsible only for `ExperienceRecord -> RewardSignal`. No repository
access, no learning, and no policy updates happen here — see
`app/reward/strategy.py` for the deterministic computation itself.
"""

from app.experience import ExperienceRecord
from app.reward.models import RewardSignal
from app.reward.strategy import BaseRewardStrategy, WeightedRewardStrategy


class RewardCalculator:
    """Converts an `ExperienceRecord` into a `RewardSignal` via an injected strategy.

    The strategy is injected via the constructor (dependency injection),
    defaulting to `WeightedRewardStrategy` — a future caller (e.g. a
    contextual-bandit or PPO training harness) can supply a different
    `BaseRewardStrategy` without any change to this class.
    """

    def __init__(self, strategy: BaseRewardStrategy | None = None) -> None:
        """Create a calculator, optionally wired to a specific strategy.

        Args:
            strategy: The `BaseRewardStrategy` to delegate to. Defaults to
                `WeightedRewardStrategy` when not provided.
        """
        self._strategy = strategy if strategy is not None else WeightedRewardStrategy()

    def calculate(self, experience: ExperienceRecord) -> RewardSignal:
        """Compute a `RewardSignal` for `experience`.

        Args:
            experience: The completed execution experience to score.

        Returns:
            The `RewardSignal` produced by this calculator's strategy.
        """
        return self._strategy.compute(experience)
