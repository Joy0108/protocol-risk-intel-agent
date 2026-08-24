"""A small state-machine engine with the LangGraph execution semantics.

Nodes are pure functions ``state -> partial_state``; the returned mapping is
merged into the state rather than replacing it. Edges are either static or
conditional on the state after the node ran. Every transition is checkpointed,
so a run can be replayed, resumed, or diffed against another run.

The reason the orchestration is a graph rather than a straight function is the
critic loop: synthesis and criticism form a cycle with a bound on it, and that
is exactly the control flow a linear pipeline cannot express cleanly.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..tracing import current

START = "__start__"
END = "__end__"

NodeFn = Callable[[dict[str, Any]], Mapping[str, Any] | None]
RouterFn = Callable[[dict[str, Any]], str]


class GraphError(RuntimeError):
    pass


@dataclass
class Checkpoint:
    step: int
    node: str
    next_node: str
    duration_ms: float
    state_keys: list[str]
    snapshot: dict[str, Any] = field(default_factory=dict)


class StateGraph:
    def __init__(self, name: str = "graph", max_steps: int = 32, keep_snapshots: bool = False):
        self.name = name
        self.max_steps = max_steps
        self.keep_snapshots = keep_snapshots
        self._nodes: dict[str, NodeFn] = {}
        self._edges: dict[str, str] = {}
        self._conditional: dict[str, tuple[RouterFn, dict[str, str]]] = {}
        self._entry: str | None = None

    # -- construction ------------------------------------------------------
    def add_node(self, name: str, fn: NodeFn) -> StateGraph:
        if name in {START, END}:
            raise GraphError(f"{name} is reserved")
        if name in self._nodes:
            raise GraphError(f"duplicate node {name!r}")
        self._nodes[name] = fn
        return self

    def add_edge(self, src: str, dst: str) -> StateGraph:
        if src == START:
            self._entry = dst
            return self
        self._edges[src] = dst
        return self

    def add_conditional_edges(self, src: str, router: RouterFn, mapping: dict[str, str]) -> StateGraph:
        self._conditional[src] = (router, mapping)
        return self

    def set_entry_point(self, name: str) -> StateGraph:
        self._entry = name
        return self

    def validate(self) -> None:
        if self._entry is None:
            raise GraphError("no entry point")
        known = set(self._nodes) | {END}
        for src, dst in self._edges.items():
            if src not in self._nodes:
                raise GraphError(f"edge from unknown node {src!r}")
            if dst not in known:
                raise GraphError(f"edge to unknown node {dst!r}")
        for src, (_router, mapping) in self._conditional.items():
            if src not in self._nodes:
                raise GraphError(f"conditional edge from unknown node {src!r}")
            for dst in mapping.values():
                if dst not in known:
                    raise GraphError(f"conditional edge to unknown node {dst!r}")
        if self._entry not in self._nodes:
            raise GraphError(f"entry point {self._entry!r} is not a node")

    # -- execution ---------------------------------------------------------
    def _next(self, node: str, state: dict[str, Any]) -> str:
        if node in self._conditional:
            router, mapping = self._conditional[node]
            key = router(state)
            if key not in mapping:
                raise GraphError(f"router for {node!r} returned unmapped key {key!r}")
            return mapping[key]
        return self._edges.get(node, END)

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        self.validate()
        tracer = current()
        state = dict(state)
        state.setdefault("_checkpoints", [])
        state.setdefault("_path", [])

        node = self._entry
        assert node is not None
        steps = 0

        with tracer.span("graph", graph=self.name) as gspan:
            while node != END:
                if steps >= self.max_steps:
                    raise GraphError(f"{self.name} exceeded max_steps={self.max_steps}; path={state['_path']}")
                fn = self._nodes.get(node)
                if fn is None:
                    raise GraphError(f"unknown node {node!r}")

                started = time.perf_counter()
                with tracer.span(f"node:{node}") as nspan:
                    update = fn(state) or {}
                    if not isinstance(update, Mapping):
                        raise GraphError(f"node {node!r} returned {type(update).__name__}, expected a mapping")
                    state.update(update)
                    nspan["attributes"]["updated"] = sorted(update.keys())

                duration = (time.perf_counter() - started) * 1000
                nxt = self._next(node, state)
                state["_path"].append(node)
                state["_checkpoints"].append(
                    Checkpoint(
                        step=steps,
                        node=node,
                        next_node=nxt,
                        duration_ms=round(duration, 3),
                        state_keys=sorted(k for k in state if not k.startswith("_")),
                        snapshot=copy.deepcopy({k: v for k, v in state.items() if not k.startswith("_")})
                        if self.keep_snapshots
                        else {},
                    )
                )
                node = nxt
                steps += 1

            gspan["attributes"]["steps"] = steps
            gspan["attributes"]["path"] = list(state["_path"])

        state["_steps"] = steps
        return state

    def to_mermaid(self) -> str:
        lines = ["graph TD", f"    {START}([start]) --> {self._entry}"]
        for src, dst in self._edges.items():
            lines.append(f"    {src} --> {dst}")
        for src, (_router, mapping) in self._conditional.items():
            for key, dst in mapping.items():
                lines.append(f"    {src} -->|{key}| {dst}")
        return "\n".join(lines)
