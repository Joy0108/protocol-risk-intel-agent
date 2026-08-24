"""Prompt-injection detection and content policy for the query channel.

Two distinct threats, handled separately:

1. **Query-channel injection** - the analyst's question tries to rewrite the
   agent's instructions. Detected here and refused before retrieval runs.
2. **Corpus-channel injection** - a retrieved audit finding or PDF page contains
   text aimed at the model. The corpus is untrusted by construction (it is
   scraped), so retrieved text is neutralised before it reaches a prompt.

The guard is a classifier over surface patterns, not a model call. That is a
deliberate choice: the defence must not itself be steerable by the input it is
defending against.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Verdict(str, Enum):
    ALLOW = "allow"
    REFUSE_INJECTION = "refuse_injection"
    REFUSE_OPERATIONAL_EXPLOIT = "refuse_operational_exploit"


@dataclass
class GuardResult:
    verdict: Verdict
    reasons: list[str]
    matched: list[str]

    @property
    def allowed(self) -> bool:
        return self.verdict is Verdict.ALLOW

    def to_dict(self) -> dict:
        return {"verdict": self.verdict.value, "reasons": self.reasons, "matched": self.matched}


_INJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"ignore\s+(all\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instruction|prompt|rule|direction)", "instruction override"),
    (r"disregard\s+(the\s+)?(previous|prior|above|retrieved|evidence|context|instruction)", "instruction override"),
    (r"\b(system\s*override|developer\s+mode|dan\s+mode|jailbreak)\b", "role escalation"),
    (r"you\s+are\s+now\s+(a|an|in)\b", "role reassignment"),
    (r"(print|reveal|repeat|show|output|leak)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions|rules)", "prompt exfiltration"),
    (r"new\s+(instruction|rule|system\s+message)s?\s*:", "instruction injection"),
    (r"answer\s+from\s+memory", "grounding bypass"),
    (r"(without|no|skip)\s+(any\s+)?(citation|source|reference)s?", "citation bypass"),
    (r"do\s+not\s+(cite|retrieve|use\s+the\s+corpus)", "grounding bypass"),
    (r"\bpretend\b.*\b(you|to be)\b", "role reassignment"),
    (r"<\s*/?\s*(system|instructions?)\s*>", "delimiter injection"),
    (r"\[\s*(system|inst|instruction)\s*\]", "delimiter injection"),
]

_EXPLOIT_PATTERNS: list[tuple[str, str]] = [
    (r"(write|give|generate|build|produce)\s+(me\s+)?(a\s+)?(ready\s*to\s*deploy|working|deployable|live)\s+(exploit|attack|drainer)", "deployable exploit request"),
    (r"\b(drain|exploit|attack|rug)\b[^.]{0,60}\b(live|mainnet|production|right now|today)\b", "live target"),
    (r"\b0x[a-fA-F0-9]{6,}\b[^.]{0,80}\b(drain|exploit|attack|steal)\b", "named live address"),
    (r"\b(drain|exploit|attack|steal from)\b[^.]{0,80}\b0x[a-fA-F0-9]{6,}\b", "named live address"),
]

_CORPUS_NEUTRALISE = [
    (re.compile(r"(?i)ignore\s+(all\s+)?previous\s+instructions?"), "[neutralised-instruction]"),
    (re.compile(r"(?i)system\s*:\s*"), "system(text): "),
    (re.compile(r"(?i)<\s*/?\s*(system|instructions?)\s*>"), "[neutralised-tag]"),
    (re.compile(r"(?i)assistant\s*:\s*"), "assistant(text): "),
]

REFUSAL_TEXT = {
    Verdict.REFUSE_INJECTION: (
        "That request tries to change my operating instructions through the query channel, "
        "which I do not accept. I answer protocol security questions from the indexed corpus "
        "and I always cite the evidence. Ask me about a vulnerability class, a contract, or an "
        "incident and I will do that."
    ),
    Verdict.REFUSE_OPERATIONAL_EXPLOIT: (
        "I will not produce a deployable exploit against a named live contract. I can explain the "
        "vulnerability class, show how the pattern was reported in past audits, and give the "
        "mitigation, which is what this corpus supports."
    ),
}


def check_query(query: str) -> GuardResult:
    reasons: list[str] = []
    matched: list[str] = []
    low = query.lower()

    for pattern, reason in _EXPLOIT_PATTERNS:
        m = re.search(pattern, low)
        if m:
            reasons.append(reason)
            matched.append(m.group(0)[:80])
    if reasons:
        return GuardResult(Verdict.REFUSE_OPERATIONAL_EXPLOIT, reasons, matched)

    for pattern, reason in _INJECTION_PATTERNS:
        m = re.search(pattern, low)
        if m:
            reasons.append(reason)
            matched.append(m.group(0)[:80])
    if reasons:
        return GuardResult(Verdict.REFUSE_INJECTION, sorted(set(reasons)), matched)

    return GuardResult(Verdict.ALLOW, [], [])


def neutralise_passage(text: str) -> str:
    """Defang instruction-shaped text inside retrieved evidence.

    Retrieved passages are wrapped as data, never as instructions, and any
    sequence that looks like a role delimiter is rewritten so the model reads it
    as quoted content.
    """
    out = text
    for pattern, replacement in _CORPUS_NEUTRALISE:
        out = pattern.sub(replacement, out)
    return out


def refusal_for(verdict: Verdict) -> str:
    return REFUSAL_TEXT.get(verdict, REFUSAL_TEXT[Verdict.REFUSE_INJECTION])
