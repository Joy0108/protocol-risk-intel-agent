"""The agent topology, declared once and executed by either engine.

The nodes, the edges, the routers and the cycle bound live here and nowhere
else. ``LangGraphAgent`` compiles this into a real ``langgraph.graph.StateGraph``;
``StateGraph`` in ``graph.py`` walks it directly with no third-party dependency.

Declaring the topology apart from its execution is what makes the conformance
test meaningful. Both engines call *the same* router functions, so when the test
asserts that a question produces an identical path, memo and citation set under
both, it is asserting that the two executors agree - not that two hand-written
copies of the graph happen to have been edited in step.

The sentinels match LangGraph's own (``__start__`` / ``__end__``) so a mapping
value can be handed to either engine unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

START = "__start__"
END = "__end__"

NodeFn = Callable[[dict[str, Any]], Mapping[str, Any] | None]
RouterFn = Callable[[dict[str, Any]], str]


class GraphError(RuntimeError):
    pass


@dataclass
class AgentSpec:
    """A declared graph: nodes, edges and routers."""

    name: str = "graph"
    entry: str | None = None
    nodes: dict[str, NodeFn] = field(default_factory=dict)
    edges: dict[str, str] = field(default_factory=dict)
    conditional: dict[str, tuple[RouterFn, dict[str, str]]] = field(default_factory=dict)
    max_steps: int = 32

    # -- construction ------------------------------------------------------
    def add_node(self, name: str, fn: NodeFn) -> AgentSpec:
        if name in {START, END}:
            raise GraphError(f"{name} is reserved")
        if name in self.nodes:
            raise GraphError(f"duplicate node {name!r}")
        self.nodes[name] = fn
        return self

    def add_edge(self, src: str, dst: str) -> AgentSpec:
        if src == START:
            self.entry = dst
            return self
        self.edges[src] = dst
        return self

    def add_conditional_edges(self, src: str, router: RouterFn, mapping: dict[str, str]) -> AgentSpec:
        self.conditional[src] = (router, mapping)
        return self

    def set_entry_point(self, name: str) -> AgentSpec:
        self.entry = name
        return self

    # -- shared semantics --------------------------------------------------
    def validate(self) -> None:
        if self.entry is None:
            raise GraphError("no entry point")
        known = set(self.nodes) | {END}
        for src, dst in self.edges.items():
            if src not in self.nodes:
                raise GraphError(f"edge from unknown node {src!r}")
            if dst not in known:
                raise GraphError(f"edge to unknown node {dst!r}")
        for src, (_router, mapping) in self.conditional.items():
            if src not in self.nodes:
                raise GraphError(f"conditional edge from unknown node {src!r}")
            for dst in mapping.values():
                if dst not in known:
                    raise GraphError(f"conditional edge to unknown node {dst!r}")
        if self.entry not in self.nodes:
            raise GraphError(f"entry point {self.entry!r} is not a node")

    def next_after(self, node: str, state: Mapping[str, Any]) -> str:
        """Where control goes after ``node``, given the post-update state.

        Both engines route through this. The routers are pure functions of
        state, so LangGraph calling one to pick an edge and this calling the
        same one to record the transition cannot disagree.
        """
        if node in self.conditional:
            router, mapping = self.conditional[node]
            key = router(dict(state))
            if key not in mapping:
                raise GraphError(f"router for {node!r} returned unmapped key {key!r}")
            return mapping[key]
        return self.edges.get(node, END)

    def to_mermaid(self) -> str:
        lines = ["graph TD", f"    {START}([start]) --> {self.entry}"]
        for src, dst in self.edges.items():
            lines.append(f"    {src} --> {dst}")
        for src, (_router, mapping) in self.conditional.items():
            for key, dst in mapping.items():
                lines.append(f"    {src} -->|{key}| {dst}")
        return "\n".join(lines)


__all__ = ["END", "START", "AgentSpec", "GraphError", "NodeFn", "RouterFn"]
