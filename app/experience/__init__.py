"""The Experience Memory Layer: records every ACRF execution as a reusable,
structured `ExperienceRecord` that future adaptive-learning algorithms can
consume.

No reinforcement learning, no contextual bandits, no Q-learning, no
PPO/DQN, no neural networks, no LLM calls, no training, no replay
buffers, no RL datasets — this module is responsible only for collecting
execution experiences.
"""

from app.experience.models import ExperienceRecord
from app.experience.recorder import ExperienceRecorder
from app.experience.repository import (
    DEFAULT_EXPERIENCE_REPOSITORY,
    ExperienceRepository,
    InMemoryExperienceRepository,
)

__all__ = [
    "DEFAULT_EXPERIENCE_REPOSITORY",
    "ExperienceRecord",
    "ExperienceRecorder",
    "ExperienceRepository",
    "InMemoryExperienceRepository",
]
