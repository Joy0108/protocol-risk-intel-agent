"""The ablation matrix.

Every row is one configuration evaluated on the same frozen golden set, so the
deltas are attributable. Rows are ordered as they were actually run: baselines
first, then one change at a time, then the combinations. Negative results stay
in the table - HyDE is in here because it lost, and a matrix that only shows the
wins is not evidence of anything.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import DEFAULT_RETRIEVAL, MANIFEST_PATH, REPORT_DIR, RetrievalConfig
from ..index.hybrid import Retriever
from .run_eval import evaluate_pages, evaluate_retrieval, load_chunks, load_golden

BASE = DEFAULT_RETRIEVAL


def matrix() -> list[RetrievalConfig]:
    return [
        # --- single retrievers ------------------------------------------------
        BASE.variant("R01 bm25-only", use_dense=False, fusion="none", rerank=False, metadata_filter=False, quantize=False, semantic_cache=False, contextual_chunks=False, mmr=False),
        BASE.variant("R02 dense-only-hash", use_bm25=False, embedder="hash", fusion="none", rerank=False, metadata_filter=False, quantize=False, semantic_cache=False, contextual_chunks=False, mmr=False),
        BASE.variant("R03 dense-only-lsa", use_bm25=False, fusion="none", rerank=False, metadata_filter=False, quantize=False, semantic_cache=False, contextual_chunks=False, mmr=False),
        # --- fusion -----------------------------------------------------------
        BASE.variant("R04 linear a=0.3", fusion="linear", linear_alpha=0.3, rerank=False, metadata_filter=False, quantize=False, semantic_cache=False, contextual_chunks=False, mmr=False),
        BASE.variant("R05 linear a=0.5", fusion="linear", linear_alpha=0.5, rerank=False, metadata_filter=False, quantize=False, semantic_cache=False, contextual_chunks=False, mmr=False),
        BASE.variant("R06 linear a=0.7", fusion="linear", linear_alpha=0.7, rerank=False, metadata_filter=False, quantize=False, semantic_cache=False, contextual_chunks=False, mmr=False),
        BASE.variant("R07 rrf k=10", rrf_k=10, rerank=False, metadata_filter=False, quantize=False, semantic_cache=False, contextual_chunks=False, mmr=False),
        BASE.variant("R08 rrf k=60", rrf_k=60, rerank=False, metadata_filter=False, quantize=False, semantic_cache=False, contextual_chunks=False, mmr=False),
        BASE.variant("R09 rrf k=120", rrf_k=120, rerank=False, metadata_filter=False, quantize=False, semantic_cache=False, contextual_chunks=False, mmr=False),
        # --- filtering and reranking -----------------------------------------
        BASE.variant("R10 rrf + metadata filter", quantize=False, semantic_cache=False, contextual_chunks=False, mmr=False),
        BASE.variant("R11 rrf + rerank", rerank=True, metadata_filter=False, quantize=False, semantic_cache=False, contextual_chunks=False, mmr=False),
        BASE.variant("R12 rrf + filter + rerank", rerank=True, quantize=False, semantic_cache=False, contextual_chunks=False, mmr=False),
        BASE.variant("R13 rerank depth 20", rerank=True, rerank_depth=20, quantize=False, semantic_cache=False, contextual_chunks=False, mmr=False),
        BASE.variant("R14 rerank depth 60", rerank=True, rerank_depth=60, quantize=False, semantic_cache=False, contextual_chunks=False, mmr=False),
        # --- indexing and diversification -------------------------------------
        BASE.variant("R15 no contextual header", contextual_chunks=False, mmr=False, quantize=False, semantic_cache=False),
        BASE.variant("R16 + contextual header", mmr=False, quantize=False, semantic_cache=False),
        BASE.variant("R17 + mmr l=0.5", mmr=True, mmr_lambda=0.5, quantize=False, semantic_cache=False),
        BASE.variant("R18 + mmr l=0.7", mmr=True, mmr_lambda=0.7, quantize=False, semantic_cache=False),
        BASE.variant("R19 + mmr l=0.9", mmr=True, mmr_lambda=0.9, quantize=False, semantic_cache=False),
        # --- query expansion --------------------------------------------------
        BASE.variant("R20 + multi-query k=2", multi_query=True, max_subqueries=2, quantize=False, semantic_cache=False),
        BASE.variant("R21 + multi-query k=4", multi_query=True, max_subqueries=4, quantize=False, semantic_cache=False),
        BASE.variant("R22 + HyDE expansion", use_hyde=True, quantize=False, semantic_cache=False),
        # --- serving ----------------------------------------------------------
        BASE.variant("R23 + int8 quantization", semantic_cache=False),
        BASE.variant("R24 final (rrf + metadata filter, quantized + cached)"),
    ]


def run_ablation(
    manifest_path: Path = MANIFEST_PATH,
    out_dir: Path = REPORT_DIR,
    write: bool = True,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    chunks = load_chunks(manifest_path)
    golden = load_golden()
    rows: list[dict[str, Any]] = []

    for cfg in matrix():
        retriever = Retriever(chunks, cfg)
        _, summary = evaluate_retrieval(retriever, golden)
        pages = evaluate_pages(retriever, golden)
        latency = retriever.latency_report()
        row = {
            "config": cfg.name,
            "ndcg@10": summary["ndcg@10"],
            "mrr": summary["mrr"],
            "recall@10": summary["recall@10"],
            "lexical_code_mrr": summary["lexical_code_mrr"],
            "page_recall@1": pages.get("late_interaction_recall@1"),
            "p50_ms": latency.get("p50_ms"),
            "p95_ms": latency.get("p95_ms"),
            "redundancy@10": summary.get("redundancy@10"),
            "vector_bytes": retriever.dense.memory_report()["stored_bytes"] if retriever.dense else 0,
        }
        rows.append(row)
        if verbose:
            print(f"  {row['config']:<32} nDCG@10 {row['ndcg@10']:.3f}  MRR {row['mrr']:.3f}  p50 {row['p50_ms']}ms")

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "ablation.json").write_text(json.dumps(rows, indent=2), encoding="utf-8", newline="\n")
        (out_dir / "ablation.md").write_text(to_markdown(rows), encoding="utf-8", newline="\n")
    return rows


def to_markdown(rows: list[dict[str, Any]]) -> str:
    headers = ["config", "ndcg@10", "mrr", "recall@10", "redundancy@10", "lexical_code_mrr", "page_recall@1", "p50_ms"]
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")

    baseline = next((r for r in rows if r["config"].startswith("R01")), None)
    final = next((r for r in rows if r["config"].startswith("R24")), None)
    hyde = next((r for r in rows if "HyDE" in r["config"]), None)
    pre_hyde = next((r for r in rows if r["config"].startswith("R16")), None)
    rerank_on = next((r for r in rows if r["config"].startswith("R11")), None)
    rerank_off = next((r for r in rows if r["config"].startswith("R08")), None)

    out.append("")
    if baseline and final:
        out.append(f"- BM25 baseline -> final: nDCG@10 {baseline['ndcg@10']} -> {final['ndcg@10']} "
                   f"(+{round(final['ndcg@10'] - baseline['ndcg@10'], 3)})")
        out.append(f"- lexical/code MRR {baseline['lexical_code_mrr']} -> {final['lexical_code_mrr']}")
    if hyde and pre_hyde:
        delta = round(hyde["ndcg@10"] - pre_hyde["ndcg@10"], 3)
        verdict = "removed" if delta <= 0 else "kept"
        out.append(f"- HyDE vs the same config without it: {delta:+} nDCG@10 -> {verdict}")
    if rerank_on and rerank_off:
        delta = round(rerank_on["ndcg@10"] - rerank_off["ndcg@10"], 3)
        verdict = "default off" if delta <= 0 else "default on"
        out.append(
            f"- feature reranker vs the same config without it: {delta:+} nDCG@10, "
            f"recall@10 {rerank_off['recall@10']} -> {rerank_on['recall@10']} -> {verdict}"
        )
    return "\n".join(out)
