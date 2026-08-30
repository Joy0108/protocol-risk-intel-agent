"""The reference engine: the same graph, walked without a dependency.

LangGraph is the production executor (``langgraph_engine.py``). This walker
exists for two reasons, and neither is nostalgia:

1. **It keeps the default install small.** ``pip install pria`` gives you the
   retrieval stack, the Solidity span extractor and the evaluation harness on
   numpy alone. Orchestration is a ``[graph]`` extra.
2. **It is the control in the conformance test.** Two independent executors
   over one declared topology, asserted to produce the same path, the same memo
   and the same citations, is a stronger statement about the agent than either
   engine passing its own tests.

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
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..tracing import current
from .spec import END, START, AgentSpec, GraphError, NodeFn, RouterFn

__all__ = [
    "END",
    "START",
    "AgentSpec",
    "Checkpoint",
    "GraphError",
    "NodeFn",
    "RouterFn",
    "StateGraph",
]


@dataclass
class Checkpoint:
    step: int
    node: str
    next_node: str
    duration_ms: float
    state_keys: list[str]
    snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step, "node": self.node, "next": self.next_node,
                "duration_ms": self.duration_ms, "state_keys": self.state_keys}


class StateGraph:
    """Walks an :class:`AgentSpec` directly."""

    engine = "reference"

    def __init__(self, name: str = "graph", max_steps: int = 32, keep_snapshots: bool = False,
                 spec: AgentSpec | None = None):
        self.spec = spec or AgentSpec(name=name, max_steps=max_steps)
        self.name = self.spec.name
        self.max_steps = self.spec.max_steps
        self.keep_snapshots = keep_snapshots

    @classmethod
    def from_spec(cls, spec: AgentSpec, keep_snapshots: bool = False) -> StateGraph:
        return cls(spec.name, spec.max_steps, keep_snapshots, spec=spec)

    # -- construction (delegates; the topology has one home) ---------------
    def add_node(self, name: str, fn: NodeFn) -> StateGraph:
        self.spec.add_node(name, fn)
        return self

    def add_edge(self, src: str, dst: str) -> StateGraph:
        self.spec.add_edge(src, dst)
        return self

    def add_conditional_edges(self, src: str, router: RouterFn, mapping: dict[str, str]) -> StateGraph:
        self.spec.add_conditional_edges(src, router, mapping)
        return self

    def set_entry_point(self, name: str) -> StateGraph:
        self.spec.set_entry_point(name)
        return self

    def validate(self) -> None:
        self.spec.validate()

    # -- execution ---------------------------------------------------------
    def invoke(self, state: Mapping[str, Any], thread_id: str | None = None) -> dict[str, Any]:
        self.validate()
        tracer = current()
        state = dict(state)
        state.setdefault("_checkpoints", [])
        state.setdefault("_path", [])

        node = self.spec.entry
        assert node is not None
        steps = 0

        with tracer.span("graph", graph=self.name, engine=self.engine):
            while node != END:
                if steps >= self.max_steps:
                    raise GraphError(f"{self.name} exceeded max_steps={self.max_steps}; path={state['_path']}")
                fn = self.spec.nodes.get(node)
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
                nxt = self.spec.next_after(node, state)
                state["_path"].append(node)
                state["_checkpoints"].append(
                    Checkpoint(
                        step=steps,
                        node=node,
                        next_node=nxt,
                        duration_ms=round(duration, 3),
                        state_keys=sorted(update),
                        snapshot=copy.deepcopy({k: v for k, v in state.items() if not k.startswith("_")})
                        if self.keep_snapshots else {},
                    )
                )
                node = nxt
                steps += 1

        state["_steps"] = steps
        state["_engine"] = self.engine
        state["_thread_id"] = thread_id
        return state

    def to_mermaid(self) -> str:
        return self.spec.to_mermaid()


def audit_trail(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The path from question to memo, as a reviewer would read it.

    Accepts either engine's checkpoints: the walker records :class:`Checkpoint`
    objects, LangGraph's instrumented nodes record plain dicts so the
    checkpointer can serialise them.
    """
    return [c.to_dict() if isinstance(c, Checkpoint) else dict(c)
            for c in state.get("_checkpoints", [])]
