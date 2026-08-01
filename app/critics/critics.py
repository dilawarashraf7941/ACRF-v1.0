"""Critic infrastructure for ACRF: the abstract critic interface and four
placeholder critic implementations.

This module contains no evaluation logic, no LLM calls, no routing, and
no adaptive policy behavior. Every concrete critic's `evaluate` method
ignores its input entirely and returns a fixed, valid, neutral
`CriticResult` — it does not inspect `content` in any way. These exist
solely to provide working, testable implementations of the `BaseCritic`
interface until real evaluation logic is implemented.
"""

from abc import ABC, abstractmethod
from typing import Any

from app.critics.models import CriticResult, CriticType


class BaseCritic(ABC):
    """Abstract interface for a critic that evaluates content and returns a `CriticResult`.

    This defines only the contract; no evaluation logic, LLM calls,
    routing, or adaptive policy behavior is implemented here. Concrete
    subclasses must set `critic_name` and `critic_type` and implement
    `evaluate`.
    """

    critic_name: str = "BaseCritic"
    critic_type: CriticType = CriticType.CUSTOM

    @abstractmethod
    def evaluate(self, content: Any) -> CriticResult:
        """Evaluate `content` and return a `CriticResult`.

        Concrete subclasses must implement the actual evaluation logic;
        no evaluation, scoring, or decision-making is implemented here.

        Args:
            content: The content to evaluate. Left untyped so this
                interface stays agnostic to any particular upstream
                representation (e.g. a `WorkerOutput`, raw text, etc.).

        Returns:
            A `CriticResult` describing the outcome of the evaluation.
        """
        raise NotImplementedError

    def _placeholder_result(self) -> CriticResult:
        """Build a fixed, neutral `CriticResult` for this critic.

        Shared by the placeholder subclasses below so each `evaluate`
        need not duplicate `CriticResult` construction. This performs no
        evaluation: it always returns the same fixed values regardless of
        any input, and does not inspect `content` at all.

        Returns:
            A `CriticResult` with `score=0.0`, `confidence=0.0`,
            `passed=None`, and a fixed placeholder `feedback` message.
        """
        return CriticResult(
            critic_name=self.critic_name,
            critic_type=self.critic_type,
            score=0.0,
            passed=None,
            confidence=0.0,
            feedback=f"{self.critic_name} is a placeholder: no evaluation logic implemented.",
            metadata={"critic_class": type(self).__name__},
        )


class LogicCritic(BaseCritic):
    """Placeholder critic for logical soundness. No evaluation logic is implemented."""

    critic_name = "LogicCritic"
    critic_type = CriticType.LOGIC

    def evaluate(self, content: Any) -> CriticResult:
        """Return a fixed placeholder `CriticResult`, ignoring `content`."""
        return self._placeholder_result()


class CodeCritic(BaseCritic):
    """Placeholder critic for source code quality. No evaluation logic is implemented."""

    critic_name = "CodeCritic"
    critic_type = CriticType.CODE

    def evaluate(self, content: Any) -> CriticResult:
        """Return a fixed placeholder `CriticResult`, ignoring `content`."""
        return self._placeholder_result()


class FactCritic(BaseCritic):
    """Placeholder critic for factual accuracy. No evaluation logic is implemented."""

    critic_name = "FactCritic"
    critic_type = CriticType.FACT

    def evaluate(self, content: Any) -> CriticResult:
        """Return a fixed placeholder `CriticResult`, ignoring `content`."""
        return self._placeholder_result()


class MetaCritic(BaseCritic):
    """Placeholder critic for reviewing other critics' results. No evaluation logic is implemented."""

    critic_name = "MetaCritic"
    critic_type = CriticType.META

    def evaluate(self, content: Any) -> CriticResult:
        """Return a fixed placeholder `CriticResult`, ignoring `content`."""
        return self._placeholder_result()
