"""`HeuristicPolicy`: the deterministic, heuristic-scoring `BasePolicy` implementation.

This is the same deterministic, feature-based scoring `policy_engine_node`
used before this refactor — via `HeuristicPolicyScorer.score` directly
against `AgentState` — now exposed through the `BasePolicy` interface. No
contextual bandit algorithm, no reinforcement learning, no policy
learning, and no reward/experience updates are implemented here; every
weight is the same fixed constant already declared in
`app/policy_engine/scorer.py`.

`select_action` internally reuses the existing scorer, ranking, and
selector rather than duplicating their logic:

- `app.policy_engine.scorer.HeuristicPolicyScorer.score_critic` — the
  exact weighted-sum formula, called directly against a `StateFeatures`
  reconstructed from `context.features` (see
  `app/context/encoder.py`'s nine `HeuristicPolicyScorer`-parity
  features, which make this reconstruction exact).
- `app.policy_engine.ranking.CriticRanking` — deterministic ordering.
- `app.policy_engine.selector.CriticSelector` — top-1 selection.
"""

from app.context import ContextVector
from app.policy.base import BasePolicy
from app.policy.models import PolicyDecision
from app.policy_engine.ranking import CriticRanking
from app.policy_engine.scorer import HeuristicPolicyScorer, StateFeatures
from app.policy_engine.selector import CriticSelector, SelectionStrategy


def _state_features_from_context(context: ContextVector) -> StateFeatures:
    """Reconstruct a `StateFeatures` from a `ContextVector`'s features.

    Reads the nine `HeuristicPolicyScorer`-parity features `ContextEncoder`
    produces (see `app/context/encoder.py`), defaulting any that are
    absent to a neutral value so this never raises on an unexpected
    `ContextVector`.

    Args:
        context: The `ContextVector` to read from.

    Returns:
        A `StateFeatures` equivalent to what
        `HeuristicPolicyScorer.extract_features` would have produced from
        the original `AgentState` this `context` was encoded from.
    """
    features = context.features
    return StateFeatures(
        uncertainty=features.get("uncertainty", 0.0),
        risk=features.get("risk", 0.0),
        task_complexity=features.get("task_complexity", 0.0),
        memory_relevance=features.get("memory_relevance", 0.0),
        requires_self_correction=bool(features.get("requires_self_correction", 0.0)),
        requires_meta_critic=bool(features.get("requires_meta_critic", 0.0)),
        is_code_output=bool(features.get("is_code_output", 0.0)),
        iteration_pressure=features.get("iteration_pressure", 0.0),
        attempt_pressure=features.get("attempt_pressure", 0.0),
    )


class HeuristicPolicy(BasePolicy):
    """Deterministic, feature-based policy — the default ACRF policy.

    Behavior is identical to the pre-refactor direct
    `HeuristicPolicyScorer().score(state, candidates)` call: the same
    weight tables, the same formula, the same deterministic alphabetical
    tie-breaking (via `CriticRanking`), and the same top-1 selection (via
    `CriticSelector`) — only the input source changed, from `AgentState`
    directly to a `ContextVector` encoding of it.
    """

    policy_name = "HeuristicPolicy"
    policy_version = "1.0.0"

    def __init__(
        self,
        scorer: HeuristicPolicyScorer | None = None,
        selection_strategy: SelectionStrategy = SelectionStrategy.TOP_1,
    ) -> None:
        """Create a policy, optionally wired to a specific scorer/strategy.

        Args:
            scorer: The `HeuristicPolicyScorer` to reuse for `score_critic`.
                Defaults to a plain `HeuristicPolicyScorer()`.
            selection_strategy: Which `CriticSelector` strategy to apply.
                Defaults to `SelectionStrategy.TOP_1`, matching this
                policy's pre-refactor behavior.
        """
        self._scorer = scorer if scorer is not None else HeuristicPolicyScorer()
        self._selection_strategy = selection_strategy

    def select_action(
        self, context: ContextVector, candidate_critics: list[str]
    ) -> PolicyDecision:
        """Score, rank, and select critics from `candidate_critics` given `context`.

        Args:
            context: The encoded observation to score against.
            candidate_critics: The critic identifiers eligible for selection.

        Returns:
            A `PolicyDecision` with `confidence` set to the top-ranked
            critic's score (or `0.0` if there are no candidates).
        """
        features = _state_features_from_context(context)
        scores = {
            critic_name: self._scorer.score_critic(critic_name, features)
            for critic_name in candidate_critics
        }
        ranking = CriticRanking(scores)
        selected_critics = CriticSelector().select(ranking, self._selection_strategy)

        top_ranked = ranking.top(1)
        confidence = top_ranked[0].score if top_ranked else 0.0

        return PolicyDecision(
            selected_critics=selected_critics,
            scores=scores,
            ranking=ranking.as_list_of_dicts(),
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            confidence=confidence,
            metadata={
                "selection_strategy": self._selection_strategy.value,
                "context_id": context.context_id,
            },
        )
