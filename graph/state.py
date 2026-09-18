"""
Custom graph executor — the state machine that drives the agent.

~30 lines of actual logic. Every line is explainable in a 4-minute
video without hand-waving about framework internals.

Each node function:
    1. Reads what it needs from AgentState
    2. Does its work
    3. Sets state["_last_branch"] to a routing key
    4. Returns the mutated state

The Graph looks up the next node from that node's `edges` dict
using the routing key. "END" terminates the run.
"""

from __future__ import annotations

from typing import Callable

from models.schemas import AgentState


class Node:
    """A single node in the agent graph."""

    def __init__(
        self,
        name: str,
        fn: Callable[[AgentState], AgentState],
        edges: dict[str, str],
    ):
        self.name = name
        self.fn = fn
        self.edges = edges


class Graph:
    """
    A minimal state-machine graph executor.

    Nodes are registered with `add_node()`. Each node declares its
    outgoing edges as a dict mapping branch keys → next node names.
    The special name "END" stops execution.
    """

    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.entry: str = ""

    def add_node(
        self,
        name: str,
        fn: Callable[[AgentState], AgentState],
        edges: dict[str, str],
    ) -> None:
        self.nodes[name] = Node(name, fn, edges)

    def set_entry(self, name: str) -> None:
        self.entry = name

    def run(self, state: AgentState, start: str | None = None) -> AgentState:
        """Execute the graph from `start` (or entry) until we reach "END"."""
        current = start or self.entry
        if not current:
            raise ValueError("No entry node set and no start node provided.")

        while current != "END":
            if current not in self.nodes:
                raise KeyError(f"Node '{current}' not found in graph.")

            node = self.nodes[current]
            print(f"\n{'─'*60}")
            print(f"  ▶ Node: {node.name}")
            print(f"{'─'*60}")

            state = node.fn(state)

            branch = state.get("_last_branch", "default")
            
            if branch in node.edges:
                current = node.edges[branch]
            elif "default" in node.edges:
                print(f"  ⚠ Branch '{branch}' not mapped in node '{node.name}', falling back to 'default'.")
                current = node.edges["default"]
            else:
                if branch in ("default", "END", ""):
                    current = "END"
                else:
                    raise KeyError(f"Node '{node.name}' returned unmapped branch '{branch}' and has no 'default' edge.")

            print(f"  ↳ Branch: {branch} → next: {current}")

        return state
