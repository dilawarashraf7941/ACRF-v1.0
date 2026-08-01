"""`OfflineEvaluator`: aggregates one `ReplayEngine`'s replayed steps into a `ReplayResult`.

Pure aggregation only — no replay mechanics (that is `ReplayEngine`'s job,
see `replay.py`) and no comparison between policies (that is
`Benchmark`'s job, see `benchmark.py`). No reinforcement learning, no
PPO, and no learning of any kind.
"""

from app.evaluation.offline.models import ReplayResult, ReplayStep
from app.evaluation.offline.replay import ReplayEngine


def _mean(values: list[float]) -> float:
    """Return the arithmetic mean of `values`, or `0.0` for an empty list."""
    return sum(values) / len(values) if values else 0.0


class OfflineEvaluator:
    """Evaluates one policy by replaying it and aggregating the results into a `ReplayResult`.

    Stateless: a single `OfflineEvaluator` instance can evaluate any
    number of policies, one `ReplayEngine` (already wired to that
    policy) per `evaluate` call.
    """

    def evaluate(self, engine: ReplayEngine, policy_name: str) -> ReplayResult:
        """Replay `engine` and aggregate the matched steps into a `ReplayResult`.

        Args:
            engine: The `ReplayEngine` (already constructed with a
                repository, a policy, and a reward calculator) to replay.
            policy_name: A label identifying the policy being evaluated,
                recorded on the returned `ReplayResult`. Independent of
                whatever `policy_name`/similar attribute the underlying
                policy object may or may not expose, so this works for
                any `ReplayablePolicy`.

        Returns:
            The aggregate `ReplayResult`. If no stored experience matched
            this policy's decisions, every numeric aggregate is `0.0` and
            `critic_selection_frequency` is empty.
        """
        steps = engine.replay()
        total_stored_experiences = engine.repository.count()
        return self._aggregate(policy_name, steps, total_stored_experiences)

    @staticmethod
    def _aggregate(
        policy_name: str, steps: list[ReplayStep], total_stored_experiences: int
    ) -> ReplayResult:
        """Fold a list of matched `ReplayStep`s into one `ReplayResult`.

        Args:
            policy_name: The label to record on the result.
            steps: The matched steps to aggregate (see `ReplayEngine.replay`).
            total_stored_experiences: `repository.count()` at replay time,
                recorded under `metadata` for context (not itself the
                denominator of any average here — `len(steps)` is).

        Returns:
            The resulting `ReplayResult`.
        """
        total_experiences = len(steps)
        rewards = [step.reward for step in steps]
        qualities = [step.quality if step.quality is not None else 0.0 for step in steps]
        iterations = [float(step.iterations) for step in steps]
        latencies = [step.latency if step.latency is not None else 0.0 for step in steps]

        total_reward = sum(rewards)
        average_reward = total_reward / total_experiences if total_experiences else 0.0
        average_quality = _mean(qualities)
        average_iterations = _mean(iterations)
        average_latency = _mean(latencies)

        frequency_counts: dict[str, int] = {}
        for step in steps:
            for critic_name in step.selected_critics:
                frequency_counts[critic_name] = frequency_counts.get(critic_name, 0) + 1
        critic_selection_frequency = (
            {name: count / total_experiences for name, count in frequency_counts.items()}
            if total_experiences
            else {}
        )

        match_rate = (
            total_experiences / total_stored_experiences if total_stored_experiences else 0.0
        )

        return ReplayResult(
            policy_name=policy_name,
            total_experiences=total_experiences,
            total_reward=total_reward,
            average_reward=average_reward,
            average_quality=average_quality,
            average_iterations=average_iterations,
            average_latency=average_latency,
            critic_selection_frequency=critic_selection_frequency,
            metadata={
                "total_stored_experiences": total_stored_experiences,
                "match_rate": match_rate,
            },
        )
