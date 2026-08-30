"""Maximal Marginal Relevance: stop the top-k being ten copies of one finding.

Relevance ranking alone has a failure mode that recall@k cannot see. This
corpus deliberately contains near-duplicates - the same reentrancy finding
written up by two audit firms, an incident described in both a post-mortem and
a report page - because deduplicating them away would hide the problem rather
than solve it. A purely relevance-ordered top-10 answers "what went wrong in
this withdraw function" with ten restatements of one answer, and the model
downstream then has ten pieces of evidence that are really one.

MMR picks results one at a time, each time maximising

    lambda * relevance(d)  -  (1 - lambda) * max_similarity(d, already_selected)

so the second copy of a finding is penalised by its similarity to the first.
``lambda = 1.0`` is exactly the relevance ordering; lower values buy diversity
at the cost of relevance, and the ablation is what decides how much is worth
buying.

Similarity uses the dense vectors when a dense index exists. When it does not -
the BM25-only ablation rows - it falls back to Jaccard overlap on tokens, so
the diversification is measurable in *every* row rather than only the ones that
happen to have embeddings.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .bm25 import tokenize


def _rank_relevance(n: int) -> np.ndarray:
    """Relevance from rank position, not from the fusion score.

    Fusion scores are not comparable across retrievers - an RRF score and a
    cosine live on different scales, and normalising them per query makes the
    trade-off with the similarity term depend on how tightly the scores happen
    to be bunched that time. Rank is stable: the top candidate scores 1.0 and
    the last scores just above 0, for every query and every fusion mode.
    """
    if n <= 1:
        return np.ones(max(n, 0), dtype=np.float32)
    return np.array([1.0 - (i / n) for i in range(n)], dtype=np.float32)


def _jaccard_matrix(texts: Sequence[str]) -> np.ndarray:
    sets = [set(tokenize(t)) for t in texts]
    n = len(sets)
    sim = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            union = sets[i] | sets[j]
            value = (len(sets[i] & sets[j]) / len(union)) if union else 0.0
            sim[i, j] = sim[j, i] = value
    return sim


def similarity_matrix(candidates: Sequence[dict[str, Any]], vectors: dict[str, np.ndarray] | None) -> np.ndarray:
    """Pairwise similarity over the candidate pool, dense if available."""
    if vectors:
        rows = [vectors.get(c["chunk_id"]) for c in candidates]
        if all(r is not None for r in rows):
            matrix = np.vstack(rows).astype(np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix = matrix / norms
            return np.clip(matrix @ matrix.T, 0.0, 1.0)
    return _jaccard_matrix([c.get("text", "") for c in candidates])


def mmr_select(
    candidates: Sequence[dict[str, Any]],
    top_k: int,
    lambda_: float = 0.7,
    vectors: dict[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    """Re-order ``candidates`` for relevance *and* non-redundancy.

    ``candidates`` must already be in relevance order. The returned list is at
    most ``top_k`` long and each result carries the diagnostics that produced
    it, so a reviewer can see *why* a passage was demoted rather than having to
    trust the ordering.
    """
    pool = list(candidates)
    if not pool or top_k <= 0:
        return []
    if lambda_ >= 1.0 or len(pool) == 1:
        return pool[:top_k]

    relevance = _rank_relevance(len(pool))
    sim = similarity_matrix(pool, vectors)

    selected: list[int] = []
    remaining = set(range(len(pool)))
    diagnostics: dict[int, dict[str, Any]] = {}

    while remaining and len(selected) < top_k:
        best_idx, best_score, best_redundancy = None, -np.inf, 0.0
        for i in sorted(remaining):
            redundancy = max((sim[i, j] for j in selected), default=0.0)
            score = lambda_ * relevance[i] - (1.0 - lambda_) * redundancy
            if score > best_score:
                best_idx, best_score, best_redundancy = i, score, redundancy
        assert best_idx is not None
        diagnostics[best_idx] = {
            "mmr_score": round(float(best_score), 6),
            "mmr_redundancy": round(float(best_redundancy), 4),
            "pre_mmr_rank": best_idx,
        }
        selected.append(best_idx)
        remaining.discard(best_idx)

    out = []
    for i in selected:
        entry = dict(pool[i])
        entry.update(diagnostics[i])
        out.append(entry)
    return out


def redundancy_at_k(results: Sequence[dict[str, Any]], vectors: dict[str, np.ndarray] | None = None) -> float:
    """Mean pairwise similarity of a result list - the thing MMR is minimising.

    Reported next to nDCG in the ablation, because a configuration that wins on
    relevance while returning near-duplicates has not necessarily produced a
    better answer, and no relevance metric will say so.
    """
    if len(results) < 2:
        return 0.0
    sim = similarity_matrix(results, vectors)
    n = len(results)
    upper = [sim[i, j] for i in range(n) for j in range(i + 1, n)]
    return round(float(np.mean(upper)), 4)


__all__ = ["mmr_select", "redundancy_at_k", "similarity_matrix"]
