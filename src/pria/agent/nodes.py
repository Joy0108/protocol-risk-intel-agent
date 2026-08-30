"""The diagnostic workflow: guard, plan, retrieve, extract, synthesise, criticise.

    START -> guard -+-> refuse ---------------------------------------> END
                    |
                    +-> plan -> retrieve -> ground -+-> abstain --------> finalise -> END
                                                    |
                                                    +-> code_analysis -> synthesise -> critic
                                                                              ^          |
                                                                              |          v
                                                                          (revise)   finalise -> END
"""

from __future__ import annotations

import re
from typing import Any

from ..config import DEFAULT_AGENT, AgentConfig
from ..index.bm25 import tokenize
from ..index.hybrid import Retriever
from ..llm import Synthesizer, build_synthesizer
from ..security.injection import Verdict, check_query, refusal_for
from ..solidity.ast_lite import extract_spans, summarise
from .critic import CitationCritic
from .graph import END, START, StateGraph
from .grounding import EvidenceGate, abstention_text
from .spec import AgentSpec, GraphError

#: ``langgraph`` when installed, otherwise the dependency-free walker.
#: ``PRIA_ENGINE`` pins it, which is what the conformance test uses to run the
#: same question through both without rebuilding the topology.
ENGINE_ENV = "PRIA_ENGINE"


def select_engine(engine: str = "auto") -> str:
    import os

    from .langgraph_engine import langgraph_available

    if engine == "auto":
        engine = os.environ.get(ENGINE_ENV, "auto")
    if engine == "auto":
        return "langgraph" if langgraph_available() else "reference"
    if engine not in {"langgraph", "reference"}:
        raise GraphError(f"unknown engine {engine!r}; expected 'langgraph', 'reference' or 'auto'")
    if engine == "langgraph" and not langgraph_available():
        raise GraphError("engine='langgraph' requested but langgraph is not installed; pip install '.[graph]'")
    return engine


def compile_agent(spec: AgentSpec, engine: str = "auto", human_in_the_loop: bool = False):
    """Turn a declared topology into something with ``.invoke``."""
    resolved = select_engine(engine)
    if resolved == "langgraph":
        from .langgraph_engine import LangGraphAgent

        return LangGraphAgent(spec, human_in_the_loop=human_in_the_loop)
    if human_in_the_loop:
        raise GraphError("the review gate needs a checkpointer; it is only available on the langgraph engine")
    return StateGraph.from_spec(spec)

_ARCHETYPE_HINTS = {
    "code_diagnostic": (r"\.sol\b", r"\bfunction\b", r"\bcontract\b", r"what is wrong with", r"identify the"),
    "incident_lookup": (r"\bexploit\b", r"\bhack\b", r"\bincident\b", r"\bpost.?mortem\b", r"\blost\b", r"\bdrained\b"),
    "mitigation": (r"\bhow should\b", r"\bcorrect way\b", r"\bmitigat", r"\bfix\b", r"\bprevent\b", r"\bmust be\b"),
    "report_page": (r"\breport\b", r"\bpage\b", r"\breview\b", r"\bappendix\b"),
}

_FILTER_HINTS = {
    "rekt.news": (r"\bincident\b", r"\bexploit(ed)?\b", r"\bhack\b", r"\bpost.?mortem\b"),
    "solidity": (r"\.sol\b",),
    "spearbit": (r"\bspearbit\b", r"\breview page\b"),
}

_CONTRACT_REF = re.compile(r"\b([A-Z][A-Za-z0-9]*\.sol)\b")


def classify(question: str) -> str:
    low = question.lower()
    for archetype, patterns in _ARCHETYPE_HINTS.items():
        if any(re.search(p, low) for p in patterns):
            return archetype
    return "vuln_pattern"


def infer_filters(question: str) -> dict[str, Any]:
    low = question.lower()
    for source, patterns in _FILTER_HINTS.items():
        if any(re.search(p, low) for p in patterns):
            return {"source": source}
    return {}


def build_agent(
    retriever: Retriever,
    synthesizer: Synthesizer | None = None,
    cfg: AgentConfig = DEFAULT_AGENT,
    corpus_root: Any = None,
    engine: str = "auto",
    human_in_the_loop: bool = False,
):
    synth = synthesizer or build_synthesizer(cfg.llm_backend, cfg.anthropic_model)
    critic = CitationCritic()
    vocabulary = {t for chunk in retriever.chunks for t in tokenize(chunk["text"])}
    gate = EvidenceGate(vocabulary, retriever._idf_table())

    # -- nodes -------------------------------------------------------------
    def guard(state: dict[str, Any]) -> dict[str, Any]:
        result = check_query(state["question"])
        return {"guard": result.to_dict(), "guard_verdict": result.verdict}

    def refuse(state: dict[str, Any]) -> dict[str, Any]:
        verdict: Verdict = state["guard_verdict"]
        return {
            "answer": refusal_for(verdict),
            "citations": [],
            "passages": [],
            "refused": True,
            "refusal_kind": verdict.value,
        }

    def plan(state: dict[str, Any]) -> dict[str, Any]:
        question = state["question"]
        return {
            "archetype": state.get("archetype") or classify(question),
            "filters": state.get("filters") if state.get("filters") is not None else infer_filters(question),
            "refused": False,
        }

    def retrieve(state: dict[str, Any]) -> dict[str, Any]:
        payload = retriever.search(state["question"], filters=state.get("filters") or None)
        return {
            "passages": payload["results"],
            "retrieval": {
                "cache_hit": payload["cache_hit"],
                "latency_ms": payload.get("latency_ms"),
                "n_lexical": payload.get("n_lexical"),
                "n_dense": payload.get("n_dense"),
            },
        }

    def ground(state: dict[str, Any]) -> dict[str, Any]:
        assessment = gate.assess(state["question"], state["passages"])
        return {"grounding": assessment.to_dict(), "grounded": assessment.sufficient, "_grounding": assessment}

    def abstain(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "answer": abstention_text(state["_grounding"], state["question"]),
            "citations": [],
            "abstained": True,
            "loops": 0,
            "critique": {"abstained": True},
            "critique_passed": True,
            "code_findings": [],
        }

    def code_analysis(state: dict[str, Any]) -> dict[str, Any]:
        """Run the AST pass over any Solidity that made it into the evidence."""
        targets = {p["doc_id"]: p for p in state["passages"] if p.get("source") == "solidity"}
        named = sorted(set(_CONTRACT_REF.findall(state["question"])))

        # When the question names a file, findings from the other contracts that
        # happened to be retrieved are noise: they are real, but they are not an
        # answer to what was asked.
        if named:
            scoped = {d: p for d, p in targets.items() if any(d.endswith(n) for n in named)}
            targets = scoped or targets

        all_spans = []
        analysed: list[str] = []

        for doc_id, passage in targets.items():
            source_path = _resolve_source(passage, corpus_root)
            source = source_path.read_text(encoding="utf-8") if source_path else passage["text"]
            all_spans.extend(extract_spans(source, doc_id))
            analysed.append(doc_id)

        # Contract spans re-scan every line their callables already cover, so
        # reporting both yields each finding twice under two different names.
        findings = [
            {"contract": span.contract, "function": span.name, "where": span.qualified_name, **hit.to_dict()}
            for span in all_spans
            if span.kind != "contract"
            for hit in span.findings
        ]
        summary = summarise(all_spans)
        summary["analysed"] = sorted(analysed)
        return {"code_findings": findings, "code_summary": summary, "named_contracts": named}

    def synthesise(state: dict[str, Any]) -> dict[str, Any]:
        critique_text = state.get("critique_prompt") or None
        draft = synth.draft(state["question"], state["passages"], critique=critique_text)
        return {
            "answer": draft.answer,
            "citations": draft.cited_ids(),
            "llm_backend": draft.backend,
            "usage": draft.usage,
            "loops": state.get("loops", 0),
        }

    def critic_node(state: dict[str, Any]) -> dict[str, Any]:
        critique = critic.review(state["answer"], state["passages"])
        return {
            "critique": critique.to_dict(),
            "critique_passed": critique.passed(cfg.min_citation_rate) if cfg.require_citations else True,
            "critique_prompt": critique.as_prompt(),
            "loops": state.get("loops", 0) + 1,
        }

    def finalise(state: dict[str, Any]) -> dict[str, Any]:
        return {"memo": _render_memo(state)}

    # -- graph -------------------------------------------------------------
    # Declared once here and compiled by whichever engine is selected.
    # max_steps doubles as LangGraph's recursion_limit, so the
    # synthesise/critic cycle is bounded identically under both.
    graph = AgentSpec("protocol-risk-agent", max_steps=24)
    for name, fn in (
        ("guard", guard),
        ("refuse", refuse),
        ("plan", plan),
        ("retrieve", retrieve),
        ("ground", ground),
        ("abstain", abstain),
        ("code_analysis", code_analysis),
        ("synthesise", synthesise),
        ("critic", critic_node),
        ("finalise", finalise),
    ):
        graph.add_node(name, fn)

    graph.add_edge(START, "guard")
    graph.add_conditional_edges(
        "guard",
        lambda s: "refuse" if s["guard_verdict"] is not Verdict.ALLOW else "continue",
        {"refuse": "refuse", "continue": "plan"},
    )
    graph.add_edge("refuse", END)
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "ground")
    graph.add_conditional_edges(
        "ground",
        lambda s: "answer" if s["grounded"] else "abstain",
        {"answer": "code_analysis", "abstain": "abstain"},
    )
    graph.add_edge("abstain", "finalise")
    graph.add_edge("code_analysis", "synthesise")
    graph.add_edge("synthesise", "critic")
    graph.add_conditional_edges(
        "critic",
        lambda s: "revise" if (not s["critique_passed"] and s["loops"] <= cfg.max_critic_loops) else "done",
        {"revise": "synthesise", "done": "finalise"},
    )
    graph.add_edge("finalise", END)
    return compile_agent(graph, engine=engine, human_in_the_loop=human_in_the_loop)


def _resolve_source(passage: dict[str, Any], corpus_root) -> Any:
    if corpus_root is None:
        return None
    rel = passage.get("metadata", {}).get("path")
    if not rel:
        return None
    candidate = corpus_root / rel
    return candidate if candidate.exists() else None


def _render_memo(state: dict[str, Any]) -> str:
    lines = [
        f"# {state['question']}",
        "",
        f"**archetype** {state.get('archetype', 'n/a')}  |  "
        f"**evidence** {len(state.get('passages', []))} passages  |  "
        f"**critic loops** {state.get('loops', 0)}",
        "",
        "## Assessment",
        state.get("answer", ""),
        "",
    ]

    findings = state.get("code_findings") or []
    if findings:
        lines += ["## Static findings from the referenced source", ""]
        for f in findings[:12]:
            lines.append(f"- `{f['where']}` line {f['line']} - {f['swc']} {f['title']} (confidence {f['confidence']})")
        lines.append("")

    passages = state.get("passages") or []
    if passages:
        lines += ["## Evidence", ""]
        for p in passages:
            label = p.get("title") or p["doc_id"]
            extra = f" ({p['severity']})" if p.get("severity") else ""
            lines.append(f"- `[cite:{p['chunk_id']}]` {p.get('source')}{extra} - {label}")
        lines.append("")

    grounding = state.get("grounding") or {}
    if grounding and not grounding.get("sufficient"):
        lines += ["## Why no answer was produced", "", f"- {grounding.get('reason')}",
                  f"- evidence coverage {grounding.get('coverage')}", ""]

    critique = state.get("critique") or {}
    if critique and not critique.get("abstained"):
        lines += [
            "## Citation check",
            "",
            f"- attribution {critique.get('attribution_rate')}  "
            f"- resolvable {critique.get('resolvable_rate')}  "
            f"- supported {critique.get('support_rate')}",
            "",
        ]
    return "\n".join(lines)


def run_question(
    retriever: Retriever,
    question: str,
    cfg: AgentConfig = DEFAULT_AGENT,
    synthesizer: Synthesizer | None = None,
    corpus_root: Any = None,
    filters: dict[str, Any] | None = None,
    engine: str = "auto",
) -> dict[str, Any]:
    graph = build_agent(retriever, synthesizer=synthesizer, cfg=cfg, corpus_root=corpus_root, engine=engine)
    return graph.invoke({"question": question, "filters": filters})
