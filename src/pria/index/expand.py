"""Query decomposition: retrieve once per facet, then fuse.

The golden set's hardest questions are not hard because the wording is
obscure. They are hard because **one question wants evidence from several
documents**:

    "How does a reentrancy attack drain a vault when the underlying token
     implements ERC777 transfer hooks?"          -> a finding + a token-standard note
    "Which incidents were caused by an oracle reading a mutable balance
     rather than a TWAP?"                        -> two post-mortems + two findings

A single embedding of that whole sentence lands between its facets and
retrieves the documents that are moderately about all of it, rather than the
documents that are decisively about each part. Recall caps out and no amount of
reranking recovers a document the first stage never returned.

Decomposition splits the query into facets, retrieves for each, and fuses the
rankings with RRF. A document that is the best answer to *one* facet reaches
the top even if it says nothing about the others - which is exactly the
behaviour a multi-document question needs.

The splitter is deterministic and domain-aware rather than a model call:

* **clause facets** - the sentence is cut at the connectives that join
  independent conditions ("when", "that", "rather than", "and", "which"),
  because that is where a compound audit question actually joins two ideas;
* **identifier facets** - token standards (``ERC777``), SWC ids and
  ``CamelCase`` contract names are pulled out on their own, since one exact
  identifier is worth more to BM25 than the sentence containing it;
* the **full query** is always kept as a facet, so decomposition can only add
  candidates, never replace the single-query ranking.

Being deterministic matters twice: the ablation row is reproducible, and a
retrieval path that needs an LLM to work cannot be the thing an LLM's answer is
grounded in.
"""

from __future__ import annotations

import re

# Connectives that join two independently answerable conditions in an audit
# question. Deliberately not a general clause splitter: commas and "of" join
# parts of one idea, and splitting on those produces fragments that retrieve
# noise.
_SPLIT = re.compile(
    r"\b(?:when|where|that|which|whilst|while|after|before|rather than|instead of|as well as|along with|and also)\b",
    re.IGNORECASE,
)

# Identifiers worth a facet of their own. An exact token like ERC4626 is the
# single most discriminative thing in a query that contains one.
_IDENTIFIER = re.compile(r"\b(?:ERC[-\s]?\d{2,4}|EIP[-\s]?\d{1,4}|SWC[-\s]?\d{3}|[A-Z][a-z]+(?:[A-Z][a-z]+)+)\b")

_MIN_FACET_TOKENS = 3


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" ,.;:?-")


def decompose(query: str, max_subqueries: int = 4) -> list[str]:
    """Split ``query`` into retrieval facets, most important first.

    The full query is always element zero. Facets are deduplicated
    case-insensitively and capped, because each one costs a retrieval pass and
    the tail of a long decomposition is mostly restatement.
    """
    full = _clean(query)
    facets: list[str] = [full]
    seen = {full.lower()}

    def add(candidate: str) -> None:
        candidate = _clean(candidate)
        if len(candidate.split()) < _MIN_FACET_TOKENS and not _IDENTIFIER.fullmatch(candidate):
            return
        if candidate.lower() in seen:
            return
        seen.add(candidate.lower())
        facets.append(candidate)

    # identifiers first: they are the highest-precision facet available
    for match in _IDENTIFIER.findall(full):
        add(match)

    for part in _SPLIT.split(full):
        add(part)

    return facets[:max_subqueries]


def fuse_rankings(rankings: list[list[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]:
    """RRF across facet rankings.

    Rank-based fusion is the right choice here for the same reason it is across
    retrievers: the sub-queries have different lengths and different score
    scales, and a document ranked first for a short identifier facet should not
    be outweighed by raw score magnitude from a long one.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


__all__ = ["decompose", "fuse_rankings"]
