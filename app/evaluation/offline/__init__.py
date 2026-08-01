"""The Offline Replay & Benchmark Framework.

Replays recorded `ExperienceRecord`s against a policy and compares
policies fairly — entirely offline, from already-stored data. No LLM
execution, no graph execution, no new experiences, no PPO, and no
reinforcement learning are implemented anywhere in this module.

- `ReplayEngine` (`replay.py`) replays every stored experience against
  one policy, using the offline "replay method" for off-policy
  evaluation of logged bandit feedback (see `replay.py`'s module
  docstring for exactly which statistical guarantees this does, and does
  not, provide for ACRF's non-randomized logging policy). Its original
  `replay` method performs no live/online learning, unchanged; its
  additional, explicitly opt-in `replay_with_learning` method
  sequentially trains the policy via `update` as it replays, so the two
  modes can be compared side by side.
- `OfflineEvaluator` (`evaluator.py`) aggregates one policy's replay into
  a `ReplayResult`.
- `Benchmark` (`benchmark.py`) compares two `ReplayResult`s into a
  `BenchmarkResult`.

This module is not wired into `app/graph/nodes.py`, `policy_engine_node`,
or `router_node`; it does not modify `app/experience`, `app/reward`,
`app/context`, or any `app/policy*` module.
"""

from app.evaluation.offline.benchmark import Benchmark
from app.evaluation.offline.evaluator import OfflineEvaluator
from app.evaluation.offline.models import BenchmarkResult, ReplayResult, ReplayStep
from app.evaluation.offline.replay import (
    DEFAULT_CANDIDATE_CRITICS,
    ReplayablePolicy,
    ReplayEngine,
    TrainablePolicy,
    build_offline_context_vector,
)

__all__ = [
    "DEFAULT_CANDIDATE_CRITICS",
    "Benchmark",
    "BenchmarkResult",
    "OfflineEvaluator",
    "ReplayEngine",
    "ReplayResult",
    "ReplayStep",
    "ReplayablePolicy",
    "TrainablePolicy",
    "build_offline_context_vector",
]
