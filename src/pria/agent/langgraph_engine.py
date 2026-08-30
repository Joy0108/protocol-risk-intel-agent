"""The diagnostic agent executed on LangGraph.

This is the production engine. It compiles the shared ``AgentSpec`` into a
``langgraph.graph.StateGraph`` and runs it under a checkpointer. Four reasons
the dependency earns its place, each replacing something this codebase would
otherwise maintain by hand:

* **Reducers put the merge rule in the type.** ``_path`` and ``_checkpoints``
  are ``Annotated[list, operator.add]``, so "a node returns a partial update
  that is merged, never substituted" stops being a convention the engine
  enforces and becomes a property of the state schema. A node could not
  overwrite the trace of what ran even if it tried.

* **The checkpointer is the replay log.** Every super-step is persisted, so a
  run can be replayed, resumed or diffed against another run - the property the
  original engine advertised, now provided durably rather than by a list this
  module appends to.

* **Interrupts are the review gate.** A memo that fails the citation critic on
  its last permitted loop is exactly the case a human should see before it
  ships. With ``human_in_the_loop=True`` the graph stops before ``finalise``
  and ``resume()`` continues from the persisted checkpoint.

* **``recursion_limit`` bounds the cycle.** ``synthesise -> critic -> synthesise``
  is a real cycle; a critic that never passes terminates with a graph error
  rather than spinning.

Span tracing is preserved: each node still runs inside a ``node:<name>`` span,
so Phoenix/OTel traces are identical under either engine.
"""

from __future__ import annotations

import operator
import time
import uuid
from collections.abc import Iterator, Mapping
from typing import Annotated, Any, TypedDict

from ..tracing import current
from .spec import END, START, AgentSpec, GraphError

#: State objects the checkpointer has to round-trip. Registered explicitly -
#: LangGraph blocks deserialising arbitrary classes out of a checkpoint, and it
#: is right to. A graph whose state can carry anything is a deserialisation
#: gadget.
ALLOWED_STATE_TYPES = [
    ("pria.security.injection", "Verdict"),
    ("pria.agent.critic", "Critique"),
    ("pria.agent.grounding", "Grounding"),
]


class AgentState(TypedDict, total=False):
    """The agent's working state.

    Everything except the two accumulators is last-write-wins, which is what
    the nodes expect: ``answer`` replaces the previous draft on a revision and
    ``loops`` counts up. The accumulators append, so the record of what ran is
    additive by construction.
    """

    question: str
    filters: dict[str, Any]
    archetype: str
    guard: dict[str, Any]
    guard_verdict: Any
    refused: bool
    refusal_kind: str
    passages: list[Any]
    retrieval: dict[str, Any]
    n_lexical: int
    n_dense: int
    cache_hit: bool
    grounded: bool
    grounding: dict[str, Any]
    _grounding: Any
    abstained: bool
    code_findings: list[Any]
    code_summary: str
    named_contracts: list[str]
    answer: str
    citations: list[str]
    source: str
    usage: dict[str, Any]
    llm_backend: str
    critique: dict[str, Any]
    critique_passed: bool
    critique_prompt: str
    loops: int
    memo: str
    latency_ms: float
    _path: Annotated[list[str], operator.add]
    _checkpoints: Annotated[list[dict[str, Any]], operator.add]


def _checkpointer() -> Any:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return MemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_STATE_TYPES))


def _assert_sentinels_match() -> None:
    """Our START/END are LangGraph's, so mapping values pass through unchanged."""
    from langgraph.graph import END as LG_END
    from langgraph.graph import START as LG_START

    if (START, END) != (LG_START, LG_END):
        raise GraphError(f"sentinel mismatch: spec uses {(START, END)}, langgraph uses {(LG_START, LG_END)}")


def _instrument(name: str, spec: AgentSpec):
    """Wrap a node so it traces itself and records its own transition.

    The wrapper keeps the two engines' checkpoint records comparable: it stores
    the same fields the reference walker stores and resolves the successor
    through ``spec.next_after`` - the very function LangGraph's conditional
    edge calls a moment later to make the same decision.
    """
    fn = spec.nodes[name]

    def node(state: AgentState) -> dict[str, Any]:
        tracer = current()
        started = time.perf_counter()
        with tracer.span(f"node:{name}") as nspan:
            update = fn(dict(state)) or {}
            if not isinstance(update, Mapping):
                raise GraphError(f"node {name!r} returned {type(update).__name__}, expected a mapping")
            update = dict(update)
            nspan["attributes"]["updated"] = sorted(update.keys())
        checkpoint = {
            "step": len(state.get("_path", [])),
            "node": name,
            "next": spec.next_after(name, {**state, **update}),
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            "state_keys": sorted(update),
        }
        return {**update, "_path": [name], "_checkpoints": [checkpoint]}

    return node


class LangGraphAgent:
    """A compiled LangGraph app behind the same interface as ``StateGraph``."""

    engine = "langgraph"

    def __init__(self, spec: AgentSpec, human_in_the_loop: bool = False):
        from langgraph.graph import StateGraph as LGStateGraph

        _assert_sentinels_match()
        spec.validate()
        self.spec = spec
        self.name = spec.name
        self.max_steps = spec.max_steps
        self.human_in_the_loop = human_in_the_loop

        builder = LGStateGraph(AgentState)
        for node_name in spec.nodes:
            builder.add_node(node_name, _instrument(node_name, spec))
        builder.add_edge(START, spec.entry)
        for src, dst in spec.edges.items():
            builder.add_edge(src, dst)
        for src, (router, mapping) in spec.conditional.items():
            builder.add_conditional_edges(src, router, mapping)

        # Pause before the memo is rendered, so a reviewer sees the answer and
        # the critique - the thing being approved - not a finished document.
        interrupt_before = ["finalise"] if human_in_the_loop and "finalise" in spec.nodes else []
        self.app = builder.compile(checkpointer=_checkpointer(), interrupt_before=interrupt_before)

    # -- execution ---------------------------------------------------------
    def _config(self, thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": self.spec.max_steps}

    def _finish(self, result: Mapping[str, Any], thread_id: str) -> dict[str, Any]:
        state = dict(result)
        state.setdefault("_path", [])
        state.setdefault("_checkpoints", [])
        state["_thread_id"] = thread_id
        state["_engine"] = self.engine
        state["_steps"] = len(state["_path"])
        return state

    def invoke(self, state: Mapping[str, Any], thread_id: str | None = None) -> dict[str, Any]:
        thread_id = thread_id or uuid.uuid4().hex[:12]
        tracer = current()
        with tracer.span("graph", graph=self.name, engine=self.engine):
            try:
                result = self.app.invoke(dict(state), config=self._config(thread_id))
            except Exception as exc:  # the recursion limit is the cycle bound
                if type(exc).__name__ == "GraphRecursionError":
                    raise GraphError(
                        f"{self.name} exceeded max_steps={self.spec.max_steps}; "
                        "the synthesise/critic cycle did not settle"
                    ) from exc
                raise
        return self._finish(result, thread_id)

    def resume(self, thread_id: str, update: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Continue a run paused at the review gate, optionally amending state."""
        config = self._config(thread_id)
        if update:
            self.app.update_state(config, dict(update))
        return self._finish(self.app.invoke(None, config=config), thread_id)

    def interrupted(self, thread_id: str) -> bool:
        return bool(self.app.get_state(self._config(thread_id)).next)

    def stream(self, state: Mapping[str, Any], thread_id: str | None = None) -> Iterator[dict[str, Any]]:
        thread_id = thread_id or uuid.uuid4().hex[:12]
        yield from self.app.stream(dict(state), config=self._config(thread_id), stream_mode="updates")

    # -- introspection -----------------------------------------------------
    def state_history(self, thread_id: str) -> list[dict[str, Any]]:
        """The framework's own checkpoint record, oldest first."""
        entries = []
        for snapshot in reversed(list(self.app.get_state_history(self._config(thread_id)))):
            path = list(snapshot.values.get("_path", []))
            entries.append({
                "step": snapshot.metadata.get("step") if snapshot.metadata else None,
                "completed": path[-1] if path else None,
                "next": list(snapshot.next),
                "path": path,
            })
        return entries

    def to_mermaid(self) -> str:
        return self.app.get_graph().draw_mermaid()

    def validate(self) -> None:
        self.spec.validate()


def langgraph_available() -> bool:
    try:
        import langgraph.graph  # noqa: F401
    except Exception:
        return False
    return True


__all__ = ["ALLOWED_STATE_TYPES", "AgentState", "LangGraphAgent", "langgraph_available"]
