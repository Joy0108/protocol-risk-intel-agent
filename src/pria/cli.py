"""Command line entry point: ``pria <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import (
    ARTIFACT_DIR,
    DEFAULT_AGENT,
    DEFAULT_RETRIEVAL,
    MANIFEST_PATH,
    RAW_DIR,
    REPORT_DIR,
    ensure_dirs,
)

# Regression gates, set below the numbers the committed report records so that
# ordinary run-to-run variation does not fail CI but a real regression does.
DEFAULT_THRESHOLDS = {
    "ndcg@10": 0.85,
    "lexical_code_mrr": 0.80,
    "resolvable_citation_rate": 0.95,
    "support_rate": 0.90,
    "adversarial_rate": 1.0,
}


def cmd_ingest(args) -> int:
    from .ingest.pipeline import run_ingest
    from .tracing import tracer

    with tracer(out_path=ARTIFACT_DIR / "spans.jsonl"):
        stats = run_ingest(raw_dir=Path(args.raw), resume=not args.rebuild, verbose=True)
    print(json.dumps(stats["manifest"], indent=2))
    return 0


def cmd_stats(args) -> int:
    from .ingest.manifest import Manifest

    with Manifest(MANIFEST_PATH) as manifest:
        stats = manifest.stats()
        dupes = manifest.duplicates()
    print(json.dumps(stats, indent=2))
    if dupes:
        print("\nduplicates:")
        for d in dupes:
            print(f"  {d['doc_id']:<16} {d['kind']:<6} of {d['duplicate_of']:<16} similarity {d['similarity']:.3f}")
    return 0


def _retriever():
    from .eval.run_eval import load_chunks
    from .index.hybrid import Retriever

    return Retriever(load_chunks(MANIFEST_PATH), DEFAULT_RETRIEVAL)


def cmd_query(args) -> int:
    retriever = _retriever()
    payload = retriever.search(args.question, top_k=args.k, explain=args.explain)
    print(f"query: {payload['query']}   cache_hit={payload['cache_hit']}   {payload['latency_ms']}ms\n")
    for i, r in enumerate(payload["results"], 1):
        print(f"{i:>2}. [{r['chunk_id']}] {r['source']:<12} score={r.get('rerank_score', r['fusion_score'])}")
        print(f"    {r.get('title')}")
        if args.explain and "rerank_features" in r:
            print(f"    features: {r['rerank_features']}")
    return 0


def cmd_ask(args) -> int:
    from .agent.nodes import run_question
    from .tracing import tracer

    ensure_dirs()
    retriever = _retriever()
    with tracer(out_path=ARTIFACT_DIR / "spans.jsonl"):
        state = run_question(retriever, args.question, cfg=DEFAULT_AGENT, corpus_root=RAW_DIR)
    print(state.get("memo") or state.get("answer", ""))
    if args.trace:
        print("\ngraph path: " + " -> ".join(state["_path"]))
    return 0


def cmd_eval(args) -> int:
    from .eval.run_eval import check_thresholds, run_full_eval

    report = run_full_eval()
    print(json.dumps({k: v for k, v in report.items() if k != "config"}, indent=2, default=str))

    thresholds = dict(DEFAULT_THRESHOLDS)
    if args.no_gate:
        return 0
    failures = check_thresholds(report, thresholds)
    if failures:
        print("\nEVAL GATE FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(f"\neval gate passed; report written to {REPORT_DIR / 'eval_report.json'}")
    return 0


def cmd_ablate(args) -> int:
    from .eval.ablation import run_ablation, to_markdown

    rows = run_ablation()
    print()
    print(to_markdown(rows))
    return 0


def cmd_graph(args) -> int:
    from .agent.nodes import build_agent

    print(build_agent(_retriever()).to_mermaid())
    return 0


def cmd_analyze(args) -> int:
    from .solidity.ast_lite import extract_spans, summarise

    source = Path(args.path).read_text(encoding="utf-8")
    spans = extract_spans(source, args.path)
    print(json.dumps(summarise(spans), indent=2))
    for span in spans:
        if span.findings:
            print(f"\n{span.qualified_name}  (lines {span.start_line}-{span.end_line})")
            for hit in span.findings:
                print(f"  line {hit.line:>3}  {hit.swc:<8} {hit.title}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pria", description="Protocol Risk and Exploit Intelligence Agent")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="build or refresh the corpus manifest")
    p.add_argument("--raw", default=str(RAW_DIR))
    p.add_argument("--rebuild", action="store_true", help="ignore the resume state and re-ingest everything")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("stats", help="show corpus statistics and detected duplicates")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("query", help="run retrieval only")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--explain", action="store_true", help="show reranker features")
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("ask", help="run the full agent and print the memo")
    p.add_argument("question")
    p.add_argument("--trace", action="store_true")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("eval", help="run the golden-set evaluation")
    p.add_argument("--no-gate", action="store_true", help="report without failing on threshold breaches")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("ablate", help="run the ablation matrix")
    p.set_defaults(func=cmd_ablate)

    p = sub.add_parser("graph", help="print the agent graph as mermaid")
    p.set_defaults(func=cmd_graph)

    p = sub.add_parser("analyze", help="run the Solidity pattern pass over one file")
    p.add_argument("path")
    p.set_defaults(func=cmd_analyze)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
