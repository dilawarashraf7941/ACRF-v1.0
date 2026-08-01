"""Graph structure definition for the ACRF LangGraph execution graph.

This module wires the placeholder nodes from `app/graph/nodes.py` and the
placeholder conditional-edge functions from `app/graph/edges.py` into a
`langgraph.graph.StateGraph` over the shared `AgentState`. It defines only
topology (which nodes exist, how they connect, where the graph starts and
may terminate) — no node or routing logic is implemented here.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph.edges import (
    CRITIC_PATH_MAP,
    ROUTER_PATH_MAP,
    SAFETY_PATH_MAP,
    SELF_CORRECTION_PATH_MAP,
    route_after_critic,
    route_after_router,
    route_after_safety,
    route_after_self_correction,
)
from app.graph.nodes import (
    NodeName,
    critic_node,
    error_feature_extractor_node,
    evaluation_node,
    planner_node,
    policy_engine_node,
    router_node,
    safety_node,
    self_correction_node,
    worker_node,
)
from app.state import AgentState


def build_graph() -> StateGraph:
    """Construct the ACRF execution graph structure (uncompiled).

    Topology:

        START -> planner -> worker -> error_feature_extractor
              -> policy_engine -> router

        router   (conditional, see `ROUTER_PATH_MAP`)
                 -> worker | critic | self_correction | safety | evaluation

        critic   (conditional, see `CRITIC_PATH_MAP`)
                 -> safety | self_correction

        self_correction (conditional, see `SELF_CORRECTION_PATH_MAP`)
                 -> worker | evaluation | END

        safety   (conditional, see `SAFETY_PATH_MAP`)
                 -> evaluation | self_correction | END

        evaluation -> END

    Every node and every conditional-edge function is an unimplemented
    placeholder; this function only declares graph shape.
    """
    graph: StateGraph = StateGraph(AgentState)

    # --- Nodes ---
    graph.add_node(NodeName.PLANNER.value, planner_node)
    graph.add_node(NodeName.WORKER.value, worker_node)
    graph.add_node(NodeName.ERROR_FEATURE_EXTRACTOR.value, error_feature_extractor_node)
    graph.add_node(NodeName.POLICY_ENGINE.value, policy_engine_node)
    graph.add_node(NodeName.ROUTER.value, router_node)
    graph.add_node(NodeName.CRITIC.value, critic_node)
    graph.add_node(NodeName.SELF_CORRECTION.value, self_correction_node)
    graph.add_node(NodeName.SAFETY.value, safety_node)
    graph.add_node(NodeName.EVALUATION.value, evaluation_node)

    # --- Entry point ---
    graph.add_edge(START, NodeName.PLANNER.value)

    # --- Fixed transitions: linear preparation pipeline before routing ---
    graph.add_edge(NodeName.PLANNER.value, NodeName.WORKER.value)
    graph.add_edge(NodeName.WORKER.value, NodeName.ERROR_FEATURE_EXTRACTOR.value)
    graph.add_edge(NodeName.ERROR_FEATURE_EXTRACTOR.value, NodeName.POLICY_ENGINE.value)
    graph.add_edge(NodeName.POLICY_ENGINE.value, NodeName.ROUTER.value)

    # --- Conditional transitions (branch selection deferred, see edges.py) ---
    graph.add_conditional_edges(NodeName.ROUTER.value, route_after_router, ROUTER_PATH_MAP)
    graph.add_conditional_edges(NodeName.CRITIC.value, route_after_critic, CRITIC_PATH_MAP)
    graph.add_conditional_edges(
        NodeName.SELF_CORRECTION.value, route_after_self_correction, SELF_CORRECTION_PATH_MAP
    )
    graph.add_conditional_edges(NodeName.SAFETY.value, route_after_safety, SAFETY_PATH_MAP)

    # --- Terminal transition ---
    graph.add_edge(NodeName.EVALUATION.value, END)

    return graph


def compile_graph() -> CompiledStateGraph:
    """Compile the graph structure returned by `build_graph`.

    Compilation only validates and freezes graph topology; it does not
    execute any node. Invoking the compiled graph will raise
    `NotImplementedError` at the first node reached, since all nodes are
    placeholders.
    """
    return build_graph().compile()
