"""`LearningAnalyzer`: derives learning-curve metrics from an already-completed
sequential replay.

Pure, read-only analysis over a `list[app.evaluation.offline.ReplayStep]`
— most meaningfully, the output of `ReplayEngine.replay_with_learning`
(see `app/evaluation/offline/replay.py`), whose per-step order reflects
a policy learning, sequentially, from the replayed log. This module
never imports `ReplayEngine`, never calls `.select_action`/`.update` on
anything, and never reads from an `ExperienceRepository` — it only ever
transforms an already-produced list of `ReplayStep`s into derived
metrics. No replay, resampling, reward computation, or policy logic is
implemented or duplicated here.

Every computation below is a pure, deterministic function of its input:
no randomness, no fitting, no learned parameters beyond simple closed-form
arithmetic (a moving average, a least-squares slope).
"""

from collections.abc import Sequence

from app.evaluation.learning_analysis.models import LearningCurve
from app.evaluation.offline import ReplayStep

DEFAULT_MOVING_AVERAGE_WINDOW = 10
"""The trailing-window size `moving_average_reward` uses by default."""

DEFAULT_CONVERGENCE_TOLERANCE = 0.05
"""The default fraction of the moving average's observed range `convergence_point` uses."""


class LearningAnalyzer:
    """Computes reward/regret/convergence metrics from a sequence of `ReplayStep`s.

    Every metric is also exposed as its own small public method (mirroring
    `app.evaluation.experiments.Analyzer`/`app.evaluation.statistics.Analyzer`'s
    established shape), in addition to the single orchestrating `analyze`
    call that builds a full `LearningCurve`.
    """

    def __init__(
        self,
        moving_average_window: int = DEFAULT_MOVING_AVERAGE_WINDOW,
        convergence_tolerance: float = DEFAULT_CONVERGENCE_TOLERANCE,
    ) -> None:
        """Create an analyzer with the given defaults.

        Args:
            moving_average_window: The default trailing-window size
                `moving_average_reward` uses when no override is passed.
            convergence_tolerance: The default fraction of the moving
                average's observed range `convergence_point` uses when
                no override is passed.

        Raises:
            ValueError: If `moving_average_window < 1` or
                `convergence_tolerance` is not strictly between 0 and 1.
        """
        if moving_average_window < 1:
            raise ValueError(f"moving_average_window must be >= 1, got {moving_average_window}")
        if not 0.0 < convergence_tolerance < 1.0:
            raise ValueError(
                "convergence_tolerance must be strictly between 0 and 1, "
                f"got {convergence_tolerance}"
            )
        self._moving_average_window = moving_average_window
        self._convergence_tolerance = convergence_tolerance

    # --- individual metrics ---------------------------------------------------------

    @staticmethod
    def reward_per_step(steps: Sequence[ReplayStep]) -> list[float]:
        """Return each step's `reward`, in order."""
        return [step.reward for step in steps]

    @staticmethod
    def cumulative_reward(rewards: Sequence[float]) -> list[float]:
        """Return the running sum of `rewards`."""
        cumulative: list[float] = []
        running = 0.0
        for reward in rewards:
            running += reward
            cumulative.append(running)
        return cumulative

    @staticmethod
    def instantaneous_regret(rewards: Sequence[float]) -> list[float]:
        """Return each step's regret relative to the best reward observed in `rewards`.

        `regret[i] = max(rewards) - rewards[i]`, always `>= 0`. This is a
        retrospective (hindsight) regret proxy: offline replay never
        observes the reward of an action it did not take, so the true
        counterfactual optimum is unknowable from replay data alone.
        Using the best reward actually observed anywhere in this
        completed run as the reference is a standard, deterministic,
        fully-computable substitute — see `README.md` for the full
        rationale and the alternative (best-so-far) this module does
        *not* use.

        Args:
            rewards: The per-step reward sequence.

        Returns:
            One regret value per input reward, or `[]` for an empty input.
        """
        if not rewards:
            return []
        best = max(rewards)
        return [best - reward for reward in rewards]

    @staticmethod
    def cumulative_regret(regrets: Sequence[float]) -> list[float]:
        """Return the running sum of `regrets` (monotonically non-decreasing)."""
        return LearningAnalyzer.cumulative_reward(regrets)

    def moving_average_reward(
        self, rewards: Sequence[float], window: int | None = None
    ) -> list[float]:
        """Return a trailing moving average of `rewards`.

        `moving_average[i]` averages `rewards[max(0, i - window + 1) : i + 1]`
        — the window shrinks near the start rather than being undefined,
        so the result is always the same length as `rewards`.

        Args:
            rewards: The per-step reward sequence.
            window: Overrides this analyzer's configured
                `moving_average_window` for this call only.

        Returns:
            One moving-average value per input reward, or `[]` for an
            empty input.

        Raises:
            ValueError: If `window` (or the configured default) is `< 1`.
        """
        effective_window = window if window is not None else self._moving_average_window
        if effective_window < 1:
            raise ValueError(f"window must be >= 1, got {effective_window}")

        rewards = list(rewards)
        result: list[float] = []
        for i in range(len(rewards)):
            start = max(0, i - effective_window + 1)
            window_slice = rewards[start : i + 1]
            result.append(sum(window_slice) / len(window_slice))
        return result

    def convergence_point(
        self, moving_average: Sequence[float], tolerance: float | None = None
    ) -> int | None:
        """Find the earliest step from which `moving_average` never again leaves a tolerance band.

        The band is `+/- tolerance * range(moving_average)` around the
        series' final value. `convergence_point` is the smallest index
        `t` such that every value from `t` to the end lies within that
        band — i.e., the earliest point after which the curve has
        effectively settled at its final level. The last index always
        trivially qualifies (it is within the band of itself), so this
        always returns a valid index for a non-empty input; it returns
        `None` only when `moving_average` is empty.

        Args:
            moving_average: Typically `moving_average_reward`'s output.
            tolerance: Overrides this analyzer's configured
                `convergence_tolerance` for this call only.

        Returns:
            The convergence step index, or `None` for an empty input.
        """
        if not moving_average:
            return None

        effective_tolerance = tolerance if tolerance is not None else self._convergence_tolerance
        moving_average = list(moving_average)
        final_value = moving_average[-1]
        value_range = max(moving_average) - min(moving_average)
        threshold = effective_tolerance * value_range if value_range > 0 else 0.0

        for start in range(len(moving_average)):
            if all(abs(value - final_value) <= threshold for value in moving_average[start:]):
                return start
        return len(moving_average) - 1  # unreachable in practice; see docstring guarantee above

    @staticmethod
    def learning_rate_estimate(rewards: Sequence[float]) -> float:
        """Estimate the reward trend's slope (reward improvement per step).

        Computed as the ordinary-least-squares slope of `rewards` against
        the step index `0, 1, 2, ...` — a simple, deterministic, standard
        summary of "how fast is performance changing over the run."
        Positive means reward tended to increase over the run; negative
        means it tended to decrease; near zero means no clear linear
        trend.

        Args:
            rewards: The per-step reward sequence.

        Returns:
            The OLS slope, or `0.0` for fewer than two rewards or a
            constant (zero-variance) step-index range (never possible in
            practice, but guarded regardless).
        """
        rewards = list(rewards)
        n = len(rewards)
        if n < 2:
            return 0.0

        steps = list(range(n))
        step_mean = sum(steps) / n
        reward_mean = sum(rewards) / n
        numerator = sum(
            (step - step_mean) * (reward - reward_mean)
            for step, reward in zip(steps, rewards, strict=True)
        )
        denominator = sum((step - step_mean) ** 2 for step in steps)
        if denominator == 0.0:
            return 0.0
        return numerator / denominator

    # --- orchestration ---------------------------------------------------------------

    def analyze(self, steps: Sequence[ReplayStep]) -> LearningCurve:
        """Compute the full `LearningCurve` for `steps`.

        Args:
            steps: A completed sequential replay's `ReplayStep`s, in the
                order they occurred (most meaningfully,
                `ReplayEngine.replay_with_learning()`'s return value).
                Never mutated.

        Returns:
            The resulting `LearningCurve`. Every field is `[]`/`0.0` for
            an empty input — no division by zero.
        """
        rewards = self.reward_per_step(steps)
        cumulative = self.cumulative_reward(rewards)
        regrets = self.instantaneous_regret(rewards)
        cumulative_regrets = self.cumulative_regret(regrets)
        moving_avg = self.moving_average_reward(rewards)
        convergence = self.convergence_point(moving_avg)
        rate = self.learning_rate_estimate(rewards)
        average = sum(rewards) / len(rewards) if rewards else 0.0

        return LearningCurve(
            reward_per_step=rewards,
            cumulative_reward=cumulative,
            instantaneous_regret=regrets,
            cumulative_regret=cumulative_regrets,
            average_reward=average,
            moving_average_reward=moving_avg,
            metadata={
                "num_steps": len(steps),
                "convergence_point": convergence,
                "learning_rate_estimate": rate,
                "moving_average_window": self._moving_average_window,
                "convergence_tolerance": self._convergence_tolerance,
                "best_reward_observed": max(rewards) if rewards else 0.0,
                "worst_reward_observed": min(rewards) if rewards else 0.0,
            },
        )
