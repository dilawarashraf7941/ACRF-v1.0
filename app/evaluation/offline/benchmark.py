"""`Benchmark`: compares two `ReplayResult`s — a baseline and a candidate policy.

Pure computation over two already-computed `ReplayResult`s — no replay
mechanics, no repository/policy/reward-calculator access, and no
learning of any kind.
"""

from app.evaluation.offline.models import BenchmarkResult, ReplayResult


class Benchmark:
    """Computes a `BenchmarkResult` comparing two `ReplayResult`s.

    Reads only the two `ReplayResult`s passed to `compare`. The winner is
    whichever policy achieved the higher `average_reward` — this
    framework's single source of truth for "which policy performed
    better," matching `app/reward`'s existing role as the definition of
    reward in ACRF.
    """

    def compare(self, baseline: ReplayResult, candidate: ReplayResult) -> BenchmarkResult:
        """Compare `candidate` against `baseline`.

        Args:
            baseline: The reference policy's `ReplayResult`.
            candidate: The policy being evaluated against `baseline`.

        Returns:
            The resulting `BenchmarkResult`. `winner` is
            `candidate.policy_name` if `candidate.average_reward >
            baseline.average_reward`, `baseline.policy_name` if the
            reverse, and `"tie"` if the two are exactly equal.
        """
        reward_improvement = candidate.average_reward - baseline.average_reward
        quality_improvement = candidate.average_quality - baseline.average_quality
        latency_difference = candidate.average_latency - baseline.average_latency
        iteration_difference = candidate.average_iterations - baseline.average_iterations

        if candidate.average_reward > baseline.average_reward:
            winner = candidate.policy_name
        elif baseline.average_reward > candidate.average_reward:
            winner = baseline.policy_name
        else:
            winner = "tie"

        return BenchmarkResult(
            baseline_policy=baseline.policy_name,
            candidate_policy=candidate.policy_name,
            reward_improvement=reward_improvement,
            quality_improvement=quality_improvement,
            latency_difference=latency_difference,
            iteration_difference=iteration_difference,
            winner=winner,
            metadata={
                "baseline": {
                    "total_experiences": baseline.total_experiences,
                    "average_reward": baseline.average_reward,
                },
                "candidate": {
                    "total_experiences": candidate.total_experiences,
                    "average_reward": candidate.average_reward,
                },
            },
        )
