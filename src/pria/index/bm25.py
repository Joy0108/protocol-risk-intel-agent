"""BM25 Okapi over the chunk store.

Kept as a first-class index rather than a baseline. Audit queries carry a lot of
rare exact tokens (``latestRoundData``, ``SWC-107``, ``get_virtual_price``) that
a 384-dimension sentence embedding blurs, and those are exactly the queries the
lexical MRR in the eval tracks.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")

_STOP = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the to was were will with
    this these those which what when where how why does do can could should would""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, plus camelCase and snake_case sub-tokens.

    ``latestRoundData`` is emitted as itself and as latest/round/data, so a query
    written either way still hits.
    """
    out: list[str] = []
    for raw in _TOKEN.findall(text):
        low = raw.lower()
        if low not in _STOP:
            out.append(low)
        parts = [p for p in re.split(r"_+", raw) if p]
        camel = [c.lower() for p in parts for c in re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|\d+", p)]
        if len(camel) > 1:
            out.extend(c for c in camel if c not in _STOP and len(c) > 1)
    return out


class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_len: list[int] = []
        self.freqs: list[Counter] = []
        self.postings: dict[str, list[int]] = defaultdict(list)
        self.df: Counter = Counter()
        self.avgdl: float = 0.0

    def build(self, records: Iterable[dict[str, Any]], text_key: str = "text", id_key: str = "chunk_id") -> BM25Index:
        for rec in records:
            toks = tokenize(rec[text_key])
            idx = len(self.doc_ids)
            self.doc_ids.append(rec[id_key])
            self.doc_len.append(len(toks))
            counts = Counter(toks)
            self.freqs.append(counts)
            for term in counts:
                self.postings[term].append(idx)
                self.df[term] += 1
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        return self

    def _idf(self, term: str) -> float:
        n = len(self.doc_ids)
        df = self.df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 10, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        terms = tokenize(query)
        if not terms:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term in set(terms):
            idf = self._idf(term)
            if idf <= 0:
                continue
            qtf = terms.count(term)
            for idx in self.postings.get(term, ()):
                if allowed is not None and self.doc_ids[idx] not in allowed:
                    continue
                tf = self.freqs[idx][term]
                denom = tf + self.k1 * (1 - self.b + self.b * (self.doc_len[idx] / self.avgdl if self.avgdl else 1.0))
                scores[idx] += idf * (tf * (self.k1 + 1) / denom) * qtf
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], self.doc_ids[kv[0]]))
        return [(self.doc_ids[i], round(s, 6)) for i, s in ranked[:top_k]]

    def __len__(self) -> int:
        return len(self.doc_ids)
