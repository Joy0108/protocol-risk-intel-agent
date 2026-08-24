"""Reranking over the fused candidate list.

Two backends:

``feature``
    A transparent scorer over query/passage features. Weights are hand-set from
    domain priors, not fitted, so it cannot overfit the golden set. This is the
    default and it is what the ablation numbers in the README measure.

``cross-encoder``
    ``cross-encoder/ms-marco-MiniLM-L-6-v2`` through sentence-transformers. The
    production path when a GPU is available; optional dependency.

Reranking runs on the top ``rerank_depth`` candidates only, so its cost is
independent of corpus size.

**It is off by default, and the ablation is the reason.** On a 73-document
corpus the candidate list after RRF is already almost pure, so there is nothing
left for a reranker to fix and its own errors dominate: compare rows R11 and R08
in ``reports/ablation.md``, which are the same pipeline with and without it, and
R11 loses on both nDCG@10 and recall@10. The mechanism is the one that matters
once a corpus is large enough that fusion returns a noisy top-40, so it stays in
the codebase behind ``rerank=True`` - but shipping it on at this scale would be
shipping a regression because the architecture diagram calls for it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

from .bm25 import tokenize

_IDENTIFIER = re.compile(r"\b(?:[a-z]+[A-Z]\w*|[A-Z]{2,}-\d+|\w+_\w+|\w+\(\))")


class FeatureReranker:
    """Interpretable relevance scorer.

    Every feature is one thing an auditor would actually look at when deciding
    whether a passage answers a question, which makes a bad ranking debuggable
    rather than mysterious.
    """

    # The fused rank is the single strongest signal, so it is weighted well
    # above any individual feature: the reranker refines the fusion ordering
    # rather than replacing it. At weight 1.0 the features outvote the prior and
    # nDCG@10 drops by 0.005 against not reranking at all.
    WEIGHTS = {
        "retrieval_prior": 4.00,
        "term_coverage": 1.60,
        "rare_term_coverage": 2.20,
        "identifier_match": 1.40,
        "title_overlap": 0.90,
        "phrase_bonus": 0.70,
        "severity_prior": 0.25,
        "length_penalty": -0.35,
    }

    def __init__(self, idf: dict[str, float] | None = None):
        self.idf = idf or {}

    def features(self, query: str, passage: dict[str, Any], prior: float) -> dict[str, float]:
        q_toks = tokenize(query)
        q_set = set(q_toks)
        p_toks = tokenize(passage.get("text", ""))
        p_set = set(p_toks)
        if not q_set:
            return {k: 0.0 for k in self.WEIGHTS}

        overlap = q_set & p_set
        coverage = len(overlap) / len(q_set)

        rare_num = sum(self.idf.get(t, 0.0) for t in overlap)
        rare_den = sum(self.idf.get(t, 0.0) for t in q_set) or 1.0

        q_ids = {m.lower() for m in _IDENTIFIER.findall(query)}
        p_low = passage.get("text", "").lower()
        identifier = (sum(1 for i in q_ids if i.strip("()") in p_low) / len(q_ids)) if q_ids else 0.0

        title_toks = set(tokenize(passage.get("title", "") or ""))
        title_overlap = (len(q_set & title_toks) / len(q_set)) if title_toks else 0.0

        bigrams = {" ".join(q_toks[i : i + 2]) for i in range(len(q_toks) - 1)}
        phrase = sum(1 for b in bigrams if b in p_low) / (len(bigrams) or 1)

        sev = {"high": 1.0, "medium": 0.5, "low": 0.2}.get((passage.get("severity") or "").lower(), 0.0)
        length_penalty = math.log1p(max(0, len(p_toks) - 160) / 100.0)

        return {
            "retrieval_prior": prior,
            "term_coverage": coverage,
            "rare_term_coverage": rare_num / rare_den,
            "identifier_match": identifier,
            "title_overlap": title_overlap,
            "phrase_bonus": phrase,
            "severity_prior": sev,
            "length_penalty": length_penalty,
        }

    def score(self, query: str, passage: dict[str, Any], prior: float) -> float:
        feats = self.features(query, passage, prior)
        return sum(self.WEIGHTS[k] * v for k, v in feats.items())

    def rerank(
        self, query: str, candidates: Sequence[dict[str, Any]], top_k: int = 10, explain: bool = False
    ) -> list[dict[str, Any]]:
        out = []
        for rank, cand in enumerate(candidates):
            prior = 1.0 / (1.0 + rank)
            item = dict(cand)
            item["rerank_score"] = round(self.score(query, cand, prior), 6)
            if explain:
                item["rerank_features"] = {k: round(v, 4) for k, v in self.features(query, cand, prior).items()}
            out.append(item)
        out.sort(key=lambda c: (-c["rerank_score"], c["chunk_id"]))
        return out[:top_k]


class CrossEncoderReranker:  # pragma: no cover - optional dependency
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: Sequence[dict[str, Any]], top_k: int = 10, explain: bool = False):
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)
        out = []
        for cand, score in zip(candidates, scores, strict=True):
            item = dict(cand)
            item["rerank_score"] = float(score)
            out.append(item)
        out.sort(key=lambda c: (-c["rerank_score"], c["chunk_id"]))
        return out[:top_k]


def build_reranker(backend: str, idf: dict[str, float] | None = None, model_name: str = ""):
    if backend == "cross-encoder":
        return CrossEncoderReranker(model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2")
    if backend == "feature":
        return FeatureReranker(idf=idf)
    raise ValueError(f"unknown rerank backend: {backend!r}")
