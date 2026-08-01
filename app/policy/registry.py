"""`PolicyRegistry`: a lookup table of `BasePolicy` instances.

Contains no policy logic of its own — only registration and lookup. This
is the mechanism by which `policy_engine_node` (see `app/graph/nodes.py`)
retrieves "the current default policy" without ever importing a concrete
policy class directly, so swapping which policy is default requires no
change to the node.
"""

from app.policy.base import BasePolicy
from app.policy.contextual_bandit_policy import ContextualBanditPolicy
from app.policy.heuristic_policy import HeuristicPolicy


class PolicyRegistry:
    """A named lookup table of `BasePolicy` instances, with one designated default.

    Policies are registered via `register`, keyed by their own
    `policy_name`. The first policy registered — or any policy registered
    with `default=True` — becomes the policy `default_policy()` returns.
    """

    def __init__(self) -> None:
        """Create an empty registry."""
        self._policies: dict[str, BasePolicy] = {}
        self._default_policy_name: str | None = None

    def register(self, policy: BasePolicy, *, default: bool = False) -> None:
        """Register `policy`, keyed by `policy.policy_name`.

        Registering a second policy under a `policy_name` already present
        replaces the first.

        Args:
            policy: The `BasePolicy` instance to register.
            default: If `True`, `policy` becomes the registry's default,
                even if another policy was previously registered as
                default. If this is the *first* policy ever registered,
                it becomes the default regardless of this flag.
        """
        self._policies[policy.policy_name] = policy
        if default or self._default_policy_name is None:
            self._default_policy_name = policy.policy_name

    def get(self, policy_name: str) -> BasePolicy:
        """Retrieve a registered policy by name.

        Args:
            policy_name: The `policy_name` it was registered under.

        Returns:
            The registered `BasePolicy` instance.

        Raises:
            KeyError: If no policy is registered under `policy_name`.
        """
        if policy_name not in self._policies:
            raise KeyError(f"No policy is registered under the name {policy_name!r}.")
        return self._policies[policy_name]

    def list(self) -> list[str]:
        """Return the names of every registered policy."""
        return list(self._policies.keys())

    def default_policy(self) -> BasePolicy:
        """Return the registry's default policy.

        Returns:
            The default `BasePolicy` instance.

        Raises:
            ValueError: If no policy has been registered yet.
        """
        if self._default_policy_name is None:
            raise ValueError("No policy has been registered; there is no default policy.")
        return self._policies[self._default_policy_name]


DEFAULT_POLICY_REGISTRY = PolicyRegistry()
"""A process-wide `PolicyRegistry`, pre-populated with the two policies this
module ships: `HeuristicPolicy` (registered as default) and
`ContextualBanditPolicy` (registered, but unusable until implemented).

`policy_engine_node` (see `app/graph/nodes.py`) is a plain function
invoked by LangGraph with only `(state)` — there is no constructor or
call site available to inject a registry into it from outside. This
module-level singleton is the practical mechanism for giving the node a
consistent, shared registry, mirroring `DEFAULT_EXPERIENCE_REPOSITORY`
and `DEFAULT_METRICS_REPOSITORY`. Tests should construct their own
`PolicyRegistry` instead of relying on this shared singleton, to stay
isolated from other tests.
"""
DEFAULT_POLICY_REGISTRY.register(HeuristicPolicy(), default=True)
DEFAULT_POLICY_REGISTRY.register(ContextualBanditPolicy())
