"""Citation-verifying critic.

Three checks, in increasing strictness:

1. **Attribution** - does every factual sentence carry a citation at all?
2. **Resolution** - does each cited id exist in the evidence that was actually
   retrieved for this question? A citation to a chunk the retriever never
   returned is a fabricated reference even when the chunk exists in the corpus.
3. **Support** - does the cited passage lexically support the sentence? This is
   the check that catches a citation attached to the wrong passage, which is the
   common failure and the one a bare id check misses entirely.

The critic returns structured problems, and the graph feeds them back into
synthesis. It never edits the draft itself, so the loop stays observable.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..index.bm25 import tokenize

_CITATION = re.compile(r"\[cite:([A-Za-z0-9_.:\-/]+)\]")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")

_HEDGE = (
    "the corpus contains no evidence",
    "i will not answer",
    "does not address this question",
    "no document in the corpus",
    "i do not accept",
    "i will not produce",
    "the model declined",
)


@dataclass
class CitationProblem:
    kind: str  # missing | unresolvable | unsupported
    sentence: str
    chunk_id: str | None = None
    support: float = 0.0

    def describe(self) -> str:
        head = self.sentence.strip()[:120]
        if self.kind == "missing":
            return f'No citation on: "{head}"'
        if self.kind == "unresolvable":
            return f'Citation [{self.chunk_id}] was not in the retrieved evidence, on: "{head}"'
        return f'Citation [{self.chunk_id}] does not support (overlap {self.support:.2f}): "{head}"'


@dataclass
class Critique:
    problems: list[CitationProblem] = field(default_factory=list)
    total_sentences: int = 0
    cited_sentences: int = 0
    total_citations: int = 0
    resolvable_citations: int = 0
    supported_citations: int = 0
    abstained: bool = False

    @property
    def attribution_rate(self) -> float:
        return self.cited_sentences / self.total_sentences if self.total_sentences else 1.0

    @property
    def resolvable_rate(self) -> float:
        return self.resolvable_citations / self.total_citations if self.total_citations else 1.0

    @property
    def support_rate(self) -> float:
        return self.supported_citations / self.total_citations if self.total_citations else 1.0

    def passed(self, threshold: float) -> bool:
        if self.abstained:
            return True
        return self.attribution_rate >= threshold and self.resolvable_rate >= threshold and self.support_rate >= threshold

    def as_prompt(self) -> str:
        if not self.problems:
            return ""
        lines = ["The previous draft has these problems:"]
        lines.extend(f"- {p.describe()}" for p in self.problems[:8])
        lines.append("Fix each one. Remove any claim you cannot support from the evidence.")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "abstained": self.abstained,
            "total_sentences": self.total_sentences,
            "cited_sentences": self.cited_sentences,
            "total_citations": self.total_citations,
            "attribution_rate": round(self.attribution_rate, 4),
            "resolvable_rate": round(self.resolvable_rate, 4),
            "support_rate": round(self.support_rate, 4),
            "problems": [{"kind": p.kind, "chunk_id": p.chunk_id, "support": round(p.support, 3)} for p in self.problems],
        }


class CitationCritic:
    def __init__(self, support_threshold: float = 0.22, min_words: int = 6):
        self.support_threshold = support_threshold
        self.min_words = min_words

    def review(self, answer: str, passages: Sequence[dict[str, Any]]) -> Critique:
        evidence = {p["chunk_id"]: p for p in passages}
        critique = Critique()

        if any(marker in answer.lower() for marker in _HEDGE) and len(answer.split()) < 90:
            critique.abstained = True
            return critique

        for sentence in _split_sentences(answer):
            if len(sentence.split()) < self.min_words:
                continue
            critique.total_sentences += 1
            cited = _CITATION.findall(sentence)
            if not cited:
                critique.problems.append(CitationProblem("missing", sentence))
                continue
            critique.cited_sentences += 1

            for chunk_id in cited:
                critique.total_citations += 1
                passage = evidence.get(chunk_id)
                if passage is None:
                    critique.problems.append(CitationProblem("unresolvable", sentence, chunk_id))
                    continue
                critique.resolvable_citations += 1
                support = _support(sentence, passage["text"])
                if support >= self.support_threshold:
                    critique.supported_citations += 1
                else:
                    critique.problems.append(CitationProblem("unsupported", sentence, chunk_id, support))

        return critique


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def _support(sentence: str, passage: str) -> float:
    """Fraction of the sentence's content terms that appear in the cited passage."""
    s_terms = {t for t in tokenize(_CITATION.sub("", sentence)) if len(t) > 2}
    if not s_terms:
        return 1.0
    p_terms = set(tokenize(passage))
    return len(s_terms & p_terms) / len(s_terms)
