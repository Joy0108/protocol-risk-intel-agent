"""Evidence-sufficiency gate.

Retrieval always returns something. On a question the corpus cannot answer it
returns the ten least-irrelevant passages, and a synthesiser handed ten passages
will write a fluent answer out of them. The gate in front of synthesis is what
turns "no evidence" into an abstention instead of a confident wrong answer.

Two independent signals, because they fail in different directions:

``unknown entity``
    The question names a proper noun the corpus has never seen - a protocol, a
    chain, an incident. High precision: if the subject of the question is absent,
    no amount of topical overlap makes the answer grounded. This is the signal
    that actually fires.

``idf-weighted coverage``
    The best passage covers too little of the question's rare vocabulary. A
    backstop for questions with no proper noun at all. The floor sits below the
    minimum coverage observed on the answerable split of the golden set, so it
    is deliberately conservative and rarely decides anything on its own.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from ..index.bm25 import tokenize

# Capitalised words that carry no entity meaning; without these, "March 2022"
# in a perfectly answerable question reads as an unknown protocol.
_NOT_ENTITIES = frozenset(
    """january february march april may june july august september october november december
    monday tuesday wednesday thursday friday saturday sunday
    which what when where why how the this that these those explain describe summarise summarize
    identify answer write give ignore system user assistant new""".split()
)

_SENTENCE_START = re.compile(r"(?:^|[.!?]\s+)([A-Z][A-Za-z0-9]*)")
_CAPITALISED = re.compile(r"\b([A-Z][A-Za-z0-9]{3,})\b")


@dataclass
class Grounding:
    sufficient: bool
    coverage: float
    unknown_entities: list[str]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sufficient": self.sufficient,
            "coverage": round(self.coverage, 4),
            "unknown_entities": self.unknown_entities,
            "reason": self.reason,
        }


class EvidenceGate:
    def __init__(self, vocabulary: Iterable[str], idf: dict[str, float], coverage_floor: float = 0.15):
        self.vocabulary = set(vocabulary)
        self.idf = idf
        self.coverage_floor = coverage_floor
        self._default_idf = max(idf.values()) if idf else 1.0

    def unknown_entities(self, question: str) -> list[str]:
        starts = {m.group(1).lower() for m in _SENTENCE_START.finditer(question)}
        out = []
        for match in _CAPITALISED.finditer(question):
            word = match.group(1)
            low = word.lower()
            if low in starts or low in _NOT_ENTITIES or low in self.vocabulary:
                continue
            if low.isdigit():
                continue
            out.append(word)
        return sorted(set(out))

    def coverage(self, question: str, passages: Sequence[dict[str, Any]], depth: int = 5) -> float:
        terms = [t for t in set(tokenize(question)) if len(t) > 3]
        if not terms:
            return 1.0
        total = sum(self.idf.get(t, self._default_idf) for t in terms) or 1.0
        best = 0.0
        for passage in passages[:depth]:
            present = set(tokenize(passage["text"]))
            covered = sum(self.idf.get(t, self._default_idf) for t in terms if t in present)
            best = max(best, covered / total)
        return best

    def assess(self, question: str, passages: Sequence[dict[str, Any]]) -> Grounding:
        if not passages:
            return Grounding(False, 0.0, [], "retrieval returned nothing")

        unknown = self.unknown_entities(question)
        cov = self.coverage(question, passages)

        if unknown:
            return Grounding(
                False,
                cov,
                unknown,
                f"the corpus contains no document mentioning {', '.join(unknown)}",
            )
        if cov < self.coverage_floor:
            return Grounding(
                False,
                cov,
                [],
                f"the best passage covers only {cov:.0%} of the question's distinctive vocabulary",
            )
        return Grounding(True, cov, [], "evidence is sufficient")


def abstention_text(grounding: Grounding, question: str) -> str:
    return (
        f"I cannot answer this from the indexed corpus: {grounding.reason}. "
        "I will not infer a figure, a date or a root cause that no retrieved document supports. "
        "Ingesting a source that covers it would make this answerable."
    )
