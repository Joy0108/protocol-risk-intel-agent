"""The retriever: three indexes, one fused ranking.

Layout of a query:

    query -> [semantic cache] -> metadata filter -> {BM25, dense} -> fusion
          -> rerank(top rerank_depth) -> top_k

The multivector page index is queried separately because pages are scored at
page granularity, not chunk granularity, and mixing the two into one ranked list
makes both metrics meaningless.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..cache import SemanticCache
from ..config import DEFAULT_RETRIEVAL, RetrievalConfig
from ..tracing import current
from .bm25 import BM25Index, tokenize
from .dense import DenseIndex
from .embed import build_embedder
from .expand import decompose, fuse_rankings
from .mmr import mmr_select, redundancy_at_k
from .multivector import CaptionIndex, MultiVectorIndex
from .rerank import build_reranker


def reciprocal_rank_fusion(rankings: Sequence[Sequence[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]:
    """RRF. Rank-based, so it needs no score calibration between retrievers."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def linear_fusion(
    lexical: Sequence[tuple[str, float]], dense: Sequence[tuple[str, float]], alpha: float = 0.5
) -> list[tuple[str, float]]:
    """Min-max normalised weighted sum. Sensitive to score distribution drift."""

    def norm(pairs: Sequence[tuple[str, float]]) -> dict[str, float]:
        if not pairs:
            return {}
        vals = [s for _, s in pairs]
        lo, hi = min(vals), max(vals)
        rng = (hi - lo) or 1.0
        return {d: (s - lo) / rng for d, s in pairs}

    lex, den = norm(lexical), norm(dense)
    keys = set(lex) | set(den)
    return sorted(
        ((k, (1 - alpha) * lex.get(k, 0.0) + alpha * den.get(k, 0.0)) for k in keys),
        key=lambda kv: (-kv[1], kv[0]),
    )


def contextualise(chunks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prepend a deterministic context header to each chunk before indexing.

    A chunk cut from an audit report keeps its words and loses its identity.
    "The balance is updated after the external call" is the same sentence in a
    high-severity reentrancy finding and in a low-severity note, and a query
    naming the protocol or the severity has nothing to match against.

    The header restates what the chunk *is* - document title, source, severity,
    SWC class, tags - so those become searchable terms on the chunk itself.
    This is the cheap, deterministic half of contextual retrieval: no model is
    called, the header is a pure function of metadata already present, and the
    index rebuild is the only cost. It is a config flag and an ablation row,
    not an assumption.

    Only the indexed text changes. ``_materialise`` still returns the original
    chunk text, so a citation quotes the document rather than the header.
    """
    out = []
    for chunk in chunks:
        parts = [
            str(chunk.get("title") or ""),
            str(chunk.get("source") or ""),
            f"severity {chunk['severity']}" if chunk.get("severity") else "",
            f"swc {chunk['swc']}" if chunk.get("swc") else "",
            " ".join(str(t) for t in chunk.get("tags", []) or []),
        ]
        header = " | ".join(p for p in parts if p)
        record = dict(chunk)
        record["text"] = f"{header}\n{chunk['text']}" if header else chunk["text"]
        out.append(record)
    return out


class Retriever:
    def __init__(self, chunks: Sequence[dict[str, Any]], cfg: RetrievalConfig = DEFAULT_RETRIEVAL):
        self.cfg = cfg
        self.chunks = list(chunks)
        self.by_id = {c["chunk_id"]: c for c in self.chunks}

        # What actually gets indexed. A chunk cut out of a report loses the
        # document identity that makes it findable - "the reentrancy one" is
        # useless without knowing which protocol and which severity. The header
        # is deterministic, so it costs an index rebuild and nothing at query
        # time, and it is an ablation row rather than an assumption.
        self.indexed = contextualise(self.chunks) if cfg.contextual_chunks else self.chunks

        self.bm25 = BM25Index(k1=cfg.bm25_k1, b=cfg.bm25_b).build(self.indexed) if cfg.use_bm25 else None

        self.dense: DenseIndex | None = None
        if cfg.use_dense:
            embedder = build_embedder(cfg.embedder, cfg.embed_dim, cfg.st_model, cfg.query_prefix)
            self.dense = DenseIndex(embedder, quantize=cfg.quantize).build(self.indexed)

        idf = self._idf_table()
        self.reranker = build_reranker(cfg.rerank_backend, idf=idf, model_name=cfg.ce_model) if cfg.rerank else None

        self.cache = SemanticCache(threshold=cfg.cache_threshold) if cfg.semantic_cache else None
        self._cache_embedder = self.dense.embedder if self.dense is not None else None

        # Page-level indexes, built lazily. Pages are reassembled from their
        # chunks first: a page is the retrieval unit here, and indexing the
        # chunks would score the same page several times under one id.
        self.pages = _reassemble_pages(self.chunks)
        self._page_index: MultiVectorIndex | None = None
        self._caption_index: CaptionIndex | None = None

        self._vector_cache: dict[str, np.ndarray] | None = None
        self.latencies_ms: list[float] = []

    # -- indexes -----------------------------------------------------------
    def _idf_table(self) -> dict[str, float]:
        if self.bm25 is not None:
            return {t: self.bm25._idf(t) for t in self.bm25.df}
        n = len(self.chunks) or 1
        df: dict[str, int] = {}
        for c in self.chunks:
            for t in set(tokenize(c["text"])):
                df[t] = df.get(t, 0) + 1
        return {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def page_index(self) -> MultiVectorIndex:
        if self._page_index is None:
            embedder = build_embedder(self.cfg.embedder, self.cfg.embed_dim, self.cfg.st_model, self.cfg.query_prefix)
            self._page_index = MultiVectorIndex(embedder).build(self.pages)
        return self._page_index

    def caption_index(self) -> CaptionIndex:
        """Single-vector baseline, built in the page index's own latent space.

        Fitting a second embedder on the captions alone would put the two
        indexes in different spaces and make the comparison meaningless, so the
        baseline reuses the projection already fitted on the page blocks.
        """
        if self._caption_index is None:
            self._caption_index = CaptionIndex(self.page_index().embedder).build(self.pages, refit=False)
        return self._caption_index

    # -- filtering ---------------------------------------------------------
    def _allowed(self, filters: dict[str, Any] | None) -> set[str] | None:
        if not filters or not self.cfg.metadata_filter:
            return None
        allowed = set()
        for c in self.chunks:
            if all(_matches(c, key, value) for key, value in filters.items()):
                allowed.add(c["chunk_id"])
        return allowed or None  # an empty filter result falls back to the whole corpus

    # -- search ------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        explain: bool = False,
    ) -> dict[str, Any]:
        top_k = top_k or self.cfg.top_k
        tracer = current()
        started = time.perf_counter()

        cache_vector = None
        if self.cache is not None and self._cache_embedder is not None:
            cache_vector = np.asarray(self._cache_embedder.encode([query], is_query=True), dtype=np.float32)[0]
            hit = self.cache.get(query, cache_vector)
            if hit is not None and hit[0].get("filters") == (filters or {}):
                cached = dict(hit[0])
                cached["cache_hit"] = True
                cached["cache_similarity"] = round(hit[1], 4)
                cached["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
                self.latencies_ms.append(cached["latency_ms"])
                return cached

        with tracer.span("retrieve", query=query[:160], config=self.cfg.name) as span:
            allowed = self._allowed(filters)
            effective_query = query
            if self.cfg.use_hyde:
                effective_query = f"{query} {hyde_expansion(query)}"

            # One retrieval pass per facet. The full query is always facet
            # zero, so decomposition can only add candidates it would otherwise
            # have missed - it never replaces the single-query ranking.
            facets = decompose(effective_query, self.cfg.max_subqueries) if self.cfg.multi_query else [effective_query]
            lexical_runs, dense_runs = [], []
            for facet in facets:
                if self.bm25:
                    lexical_runs.append(self.bm25.search(facet, self.cfg.candidate_k, allowed))
                if self.dense:
                    dense_runs.append(self.dense.search(facet, self.cfg.candidate_k, allowed))
            lexical = fuse_rankings(lexical_runs, self.cfg.rrf_k) if len(lexical_runs) > 1 else (lexical_runs[0] if lexical_runs else [])
            densehits = fuse_rankings(dense_runs, self.cfg.rrf_k) if len(dense_runs) > 1 else (dense_runs[0] if dense_runs else [])

            if self.cfg.fusion == "rrf":
                fused = reciprocal_rank_fusion([r for r in (lexical, densehits) if r], k=self.cfg.rrf_k)
            elif self.cfg.fusion == "linear":
                fused = linear_fusion(lexical, densehits, alpha=self.cfg.linear_alpha)
            else:
                fused = list(densehits or lexical)

            # The pool MMR chooses from. Selecting k out of k is a reordering;
            # diversification needs candidates the relevance ranking would have
            # dropped, so the pool is oversampled whenever MMR is on.
            if self.reranker is not None:
                depth = self.cfg.rerank_depth
            elif self.cfg.mmr:
                depth = max(self.cfg.mmr_depth, top_k)
            else:
                depth = top_k
            candidates = [self._materialise(cid, score) for cid, score in fused[:depth] if cid in self.by_id]

            if self.reranker is not None:
                ranked = self.reranker.rerank(query, candidates, top_k=len(candidates), explain=explain)
            else:
                ranked = candidates

            # Diversify last, over the ranked pool. Doing it before reranking
            # would diversify candidates the reranker was about to discard.
            if self.cfg.mmr:
                results = mmr_select(ranked, top_k, self.cfg.mmr_lambda, self._dense_vectors())
            else:
                results = ranked[:top_k]

            span["attributes"].update(
                {"lexical": len(lexical), "dense": len(densehits), "fused": len(fused),
                 "facets": len(facets), "returned": len(results)}
            )

        payload = {
            "query": query,
            "filters": filters or {},
            "results": results,
            "cache_hit": False,
            "config": self.cfg.name,
            "n_lexical": len(lexical),
            "n_dense": len(densehits),
            "facets": facets,
            "redundancy": redundancy_at_k(results, self._dense_vectors()),
        }
        if self.cache is not None and cache_vector is not None:
            self.cache.put(query, cache_vector, payload)
        payload["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        self.latencies_ms.append(payload["latency_ms"])
        return payload

    def _dense_vectors(self) -> dict[str, np.ndarray] | None:
        """chunk_id -> embedding, for MMR's similarity term."""
        if self.dense is None or self.dense.vectors is None:
            return None
        if self._vector_cache is None:
            self._vector_cache = dict(zip(self.dense.ids, self.dense.vectors, strict=True))
        return self._vector_cache

    def _materialise(self, chunk_id: str, score: float) -> dict[str, Any]:
        c = self.by_id[chunk_id]
        return {
            "chunk_id": chunk_id,
            "doc_id": c["doc_id"],
            "text": c["text"],
            "title": c.get("title"),
            "source": c.get("source"),
            "severity": c.get("severity"),
            "swc": c.get("swc"),
            "tags": c.get("tags", []),
            "metadata": c.get("metadata", {}),
            "chunk_metadata": c.get("chunk_metadata", {}),
            "fusion_score": round(float(score), 6),
        }

    # -- reporting ---------------------------------------------------------
    def latency_report(self) -> dict[str, Any]:
        if not self.latencies_ms:
            return {"n": 0}
        arr = np.array(self.latencies_ms)
        return {
            "n": int(arr.size),
            "p50_ms": round(float(np.percentile(arr, 50)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "mean_ms": round(float(arr.mean()), 2),
            "cache": self.cache.stats.to_dict() if self.cache else None,
        }


def _matches(chunk: dict[str, Any], key: str, value: Any) -> bool:
    haystack = chunk.get(key)
    if haystack is None:
        haystack = chunk.get("metadata", {}).get(key)
    if haystack is None:
        return False
    if isinstance(haystack, list):
        return value in haystack
    if isinstance(value, (list, tuple, set)):
        return haystack in value
    return str(haystack).lower() == str(value).lower()


def hyde_expansion(query: str) -> str:
    """Hypothetical document expansion.

    Retained so the ablation row that removed it stays runnable. On this corpus
    it hurts: audit queries are already dense with the exact identifiers that
    appear in the target passage, and a generated pseudo-answer dilutes them
    with generic security vocabulary that matches every finding equally well.
    """
    toks = [t for t in tokenize(query) if len(t) > 3]
    template = (
        "The finding describes a vulnerability in a smart contract. "
        "The root cause is a missing check in the affected function. "
        "An attacker can exploit this to extract value from the protocol. "
        "The recommended mitigation is to validate the state before the external call. "
    )
    return template + " ".join(toks)


def _reassemble_pages(chunks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join the chunks of each report page back into one page record."""
    pages: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        if chunk.get("metadata", {}).get("kind") != "report_page":
            continue
        entry = pages.setdefault(
            chunk["doc_id"],
            {"doc_id": chunk["doc_id"], "title": chunk.get("title"), "metadata": chunk.get("metadata", {}), "_parts": []},
        )
        entry["_parts"].append((chunk["ordinal"], chunk["text"]))
    out = []
    for entry in pages.values():
        entry["text"] = " ".join(text for _, text in sorted(entry.pop("_parts")))
        out.append(entry)
    return sorted(out, key=lambda p: p["doc_id"])
