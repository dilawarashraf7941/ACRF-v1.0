"""The Context Encoding Layer: converts `AgentState` (and, optionally, a
matching `ExperienceRecord`) into a deterministic, numeric `ContextVector`
that a future learning policy can consume as-is.

No reinforcement learning, no contextual bandits, no policy optimization,
and no learning of any kind — every feature and every normalization bound
is a fixed constant chosen at implementation time. Nothing here is fitted
from a batch of data. Must work unchanged for the current Heuristic
Policy and for any future Contextual Bandit / Offline RL / PPO /
Q-learning policy.
"""

from app.context.encoder import (
    EXECUTION_STATUS_CODES,
    SAFETY_STATUS_CODES,
    UNRECOGNIZED_STATUS_CODE,
    ContextEncoder,
)
from app.context.models import ContextVector
from app.context.normalizer import FEATURE_BOUNDS, NORMALIZATION_STRATEGY_NAME, ContextNormalizer

__all__ = [
    "EXECUTION_STATUS_CODES",
    "FEATURE_BOUNDS",
    "NORMALIZATION_STRATEGY_NAME",
    "SAFETY_STATUS_CODES",
    "UNRECOGNIZED_STATUS_CODE",
    "ContextEncoder",
    "ContextNormalizer",
    "ContextVector",
]
