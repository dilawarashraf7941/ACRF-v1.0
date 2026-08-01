"""The Reward Engine: converts completed execution experiences into
deterministic reward signals for future adaptive-learning algorithms to
consume.

No reinforcement learning, no contextual bandits, no PPO/DQN/Q-learning,
no neural networks, no replay buffers, no policy optimization, and no
LLM calls — this module is responsible only for scoring already-completed
`ExperienceRecord`s.
"""

from app.reward.calculator import RewardCalculator
from app.reward.models import RewardSignal
from app.reward.strategy import BaseRewardStrategy, WeightedRewardStrategy

__all__ = [
    "BaseRewardStrategy",
    "RewardCalculator",
    "RewardSignal",
    "WeightedRewardStrategy",
]
