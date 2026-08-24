"""Retrieval metrics with graded relevance.

Binary relevance is the wrong instrument here. A question about read-only
reentrancy has one finding that answers it and two that are genuinely useful
context, and a ranking that puts the context first is worse but not wrong. The
golden set therefore carries primary (gain 3) and secondary (gain 1) labels, and
nDCG is computed against those gains.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

PRIMARY_GAIN = 3.0
SECONDARY_GAIN = 1.0


def gains_for(doc_ids: Sequence[str], primary: Iterable[str], secondary: Iterable[str]) -> list[float]:
    p, s = set(primary), set(secondary)
    return [PRIMARY_GAIN if d in p else (SECONDARY_GAIN if d in s else 0.0) for d in doc_ids]


def dcg(gains: Sequence[float], k: int) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(ranked_docs: Sequence[str], primary: Iterable[str], secondary: Iterable[str], k: int = 10) -> float:
    primary, secondary = list(primary), list(secondary)
    if not primary and not secondary:
        return float("nan")
    gains = gains_for(ranked_docs, primary, secondary)
    ideal = sorted([PRIMARY_GAIN] * len(set(primary)) + [SECONDARY_GAIN] * len(set(secondary)), reverse=True)
    denom = dcg(ideal, k)
    return (dcg(gains, k) / denom) if denom else float("nan")


def reciprocal_rank(ranked_docs: Sequence[str], relevant: Iterable[str]) -> float:
    rel = set(relevant)
    if not rel:
        return float("nan")
    for i, doc in enumerate(ranked_docs, start=1):
        if doc in rel:
            return 1.0 / i
    return 0.0


def recall_at_k(ranked_docs: Sequence[str], relevant: Iterable[str], k: int = 10) -> float:
    rel = set(relevant)
    if not rel:
        return float("nan")
    return len(rel & set(ranked_docs[:k])) / len(rel)


def precision_at_k(ranked_docs: Sequence[str], relevant: Iterable[str], k: int = 10) -> float:
    rel = set(relevant)
    if not k:
        return float("nan")
    return len([d for d in ranked_docs[:k] if d in rel]) / k


def hit_at_k(ranked_docs: Sequence[str], relevant: Iterable[str], k: int = 3) -> float:
    rel = set(relevant)
    if not rel:
        return float("nan")
    return 1.0 if rel & set(ranked_docs[:k]) else 0.0


def mean(values: Iterable[float]) -> float:
    vals = [v for v in values if v == v]  # drop NaN
    return sum(vals) / len(vals) if vals else float("nan")


def dedupe_preserving_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def summarise_metrics(rows: Sequence[dict]) -> dict[str, float]:
    keys = {k for row in rows for k, v in row.items() if isinstance(v, (int, float))}
    return {k: round(mean(row.get(k, float("nan")) for row in rows), 4) for k in sorted(keys)}
