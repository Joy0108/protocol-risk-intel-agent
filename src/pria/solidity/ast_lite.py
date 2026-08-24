"""Solidity span extraction and vulnerability pattern matching.

Retrieval over source code fails badly when the unit is a fixed token window,
because the window rarely lines up with a function. This module cuts the file
into declaration-aligned spans (contract, function, modifier, constructor) so a
retrieved hit is always a whole callable with its signature attached.

Backend: ``tree_sitter_solidity`` is used when installed. The brace-matching
fallback below handles the same span set and is what CI runs, so the default
install has no native build step.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

_CONTRACT = re.compile(r"^\s*(contract|library|interface|abstract\s+contract)\s+([A-Za-z_]\w*)", re.M)
_CALLABLE = re.compile(
    r"^\s*(function\s+([A-Za-z_]\w*)|constructor|modifier\s+([A-Za-z_]\w*)|receive|fallback)\b",
    re.M,
)
_PRAGMA = re.compile(r"^\s*pragma\s+solidity\s+([^;]+);", re.M)


@dataclass
class CodeSpan:
    """A declaration-aligned region of a Solidity file."""

    name: str
    kind: str  # contract | function | modifier | constructor | receive | fallback
    contract: str
    start_line: int
    end_line: int
    signature: str
    text: str
    findings: list[PatternHit] = field(default_factory=list)

    @property
    def qualified_name(self) -> str:
        return f"{self.contract}.{self.name}" if self.contract and self.kind != "contract" else self.name

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "contract": self.contract,
            "qualified_name": self.qualified_name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "signature": self.signature,
            "findings": [f.to_dict() for f in self.findings],
        }


@dataclass
class PatternHit:
    rule_id: str
    swc: str
    title: str
    line: int
    evidence: str
    confidence: float

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "swc": self.swc,
            "title": self.title,
            "line": self.line,
            "evidence": self.evidence.strip()[:200],
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
# span extraction
# ---------------------------------------------------------------------------

def _strip_for_scan(source: str) -> str:
    """Blank out string literals and comments so brace counting is not fooled.

    Characters are replaced one for one, keeping every offset and line number
    identical to the original text.
    """
    out = list(source)
    i, n = 0, len(source)
    while i < n:
        ch = source[i]
        if ch == "/" and i + 1 < n and source[i + 1] == "/":
            while i < n and source[i] != "\n":
                out[i] = " "
                i += 1
        elif ch == "/" and i + 1 < n and source[i + 1] == "*":
            while i < n and not (source[i] == "*" and i + 1 < n and source[i + 1] == "/"):
                if source[i] != "\n":
                    out[i] = " "
                i += 1
            for j in range(i, min(i + 2, n)):
                out[j] = " "
            i += 2
        elif ch in "\"'":
            quote = ch
            out[i] = " "
            i += 1
            while i < n and source[i] != quote:
                if source[i] == "\\":
                    out[i] = " "
                    i += 1
                    if i < n:
                        out[i] = " "
                        i += 1
                    continue
                if source[i] != "\n":
                    out[i] = " "
                i += 1
            if i < n:
                out[i] = " "
                i += 1
        else:
            i += 1
    return "".join(out)


def _match_block(scan: str, start: int) -> tuple[int, int] | None:
    """Return (open_brace_index, index_after_close) for the block after ``start``."""
    depth = 0
    open_idx = None
    for i in range(start, len(scan)):
        c = scan[i]
        if c == ";" and open_idx is None:
            return None  # declaration without a body (interface / abstract)
        if c == "{":
            if open_idx is None:
                open_idx = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and open_idx is not None:
                return open_idx, i + 1
    return None


def extract_spans(source: str, path: str = "<memory>") -> list[CodeSpan]:
    """Cut ``source`` into contract and callable spans."""
    try:  # pragma: no cover - exercised only when the native grammar is present
        return _tree_sitter_spans(source, path)
    except Exception:
        pass

    scan = _strip_for_scan(source)
    line_of = _line_index(source)
    spans: list[CodeSpan] = []

    contracts: list[tuple[str, int, int]] = []
    for m in _CONTRACT.finditer(scan):
        block = _match_block(scan, m.end())
        if not block:
            continue
        open_idx, close_idx = block
        name = m.group(2)
        contracts.append((name, open_idx, close_idx))
        spans.append(
            CodeSpan(
                name=name,
                kind="contract",
                contract=name,
                start_line=line_of(m.start()),
                end_line=line_of(close_idx - 1),
                signature=source[m.start() : open_idx].strip(),
                text=source[m.start() : close_idx],
            )
        )

    for m in _CALLABLE.finditer(scan):
        block = _match_block(scan, m.end())
        if not block:
            continue
        open_idx, close_idx = block
        head = m.group(1)
        if head.startswith("function"):
            kind, name = "function", m.group(2)
        elif head.startswith("modifier"):
            kind, name = "modifier", m.group(3)
        else:
            kind = name = head.strip()
        owner = next((c for c, o, cl in contracts if o < m.start() < cl), "")
        spans.append(
            CodeSpan(
                name=name,
                kind=kind,
                contract=owner,
                start_line=line_of(m.start()),
                end_line=line_of(close_idx - 1),
                signature=" ".join(source[m.start() : open_idx].split()),
                text=source[m.start() : close_idx],
            )
        )

    for span in spans:
        span.findings = detect_patterns(span.text, base_line=span.start_line)
    return spans


def _line_index(source: str):
    starts = [0]
    for i, ch in enumerate(source):
        if ch == "\n":
            starts.append(i + 1)

    def line_of(idx: int) -> int:
        lo, hi = 0, len(starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if starts[mid] <= idx:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    return line_of


def _tree_sitter_spans(source: str, path: str) -> list[CodeSpan]:  # pragma: no cover
    import tree_sitter_solidity  # type: ignore
    from tree_sitter import Language, Parser  # type: ignore

    parser = Parser(Language(tree_sitter_solidity.language()))
    tree = parser.parse(source.encode())
    wanted = {
        "contract_declaration": "contract",
        "interface_declaration": "contract",
        "library_declaration": "contract",
        "function_definition": "function",
        "modifier_definition": "modifier",
        "constructor_definition": "constructor",
        "fallback_receive_definition": "fallback",
    }
    spans: list[CodeSpan] = []
    stack = [tree.root_node]
    current_contract = ""
    while stack:
        node = stack.pop()
        kind = wanted.get(node.type)
        if kind:
            name_node = node.child_by_field_name("name")
            name = source[name_node.start_byte : name_node.end_byte] if name_node else node.type
            if kind == "contract":
                current_contract = name
            text = source[node.start_byte : node.end_byte]
            spans.append(
                CodeSpan(
                    name=name,
                    kind=kind,
                    contract=current_contract,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    signature=" ".join(text.split("{")[0].split()),
                    text=text,
                    findings=detect_patterns(text, base_line=node.start_point[0] + 1),
                )
            )
        stack.extend(node.children)
    if not spans:
        raise RuntimeError("tree-sitter produced no spans")
    return spans


# ---------------------------------------------------------------------------
# vulnerability pattern extraction
# ---------------------------------------------------------------------------

_STATE_WRITE = re.compile(r"\b([A-Za-z_]\w*)\s*(\[[^\]]*\])?\s*(=|\+=|-=)\s*[^=]")
_EXTERNAL_CALL = re.compile(r"\.(transfer|transferFrom|send|call|delegatecall|safeTransfer|safeTransferFrom)\b")


def detect_patterns(text: str, base_line: int = 1) -> list[PatternHit]:
    """Static pattern pass over one span. Deliberately high recall.

    These are heuristics, not proofs. The agent treats every hit as a lead that
    must be corroborated by a retrieved audit finding before it appears in a
    memo, which is what keeps the false positives out of the output.
    """
    hits: list[PatternHit] = []
    lines = text.splitlines()

    def add(rule: str, swc: str, title: str, idx: int, conf: float) -> None:
        hits.append(PatternHit(rule, swc, title, base_line + idx, lines[idx] if idx < len(lines) else "", conf))

    call_line = None
    for i, raw in enumerate(lines):
        line = raw.split("//")[0]
        low = line.lower()

        if "tx.origin" in line:
            add("tx-origin-auth", "SWC-115", "Authorisation compares tx.origin instead of msg.sender", i, 0.95)

        if "ecrecover(" in line:
            window = "\n".join(lines[i : i + 6])
            if "address(0)" not in window and "!= address" not in window:
                add("ecrecover-unchecked", "SWC-117", "ecrecover result not checked against the zero address", i, 0.8)

        if "getreserves()" in low:
            add("spot-price-reserves", "SWC-000", "Price derived from mutable AMM reserves is manipulable in one transaction", i, 0.85)

        if "latestrounddata()" in low:
            window = "\n".join(lines[i : i + 8])
            if "updatedAt" not in window and "updatedat" not in window.lower():
                add("oracle-staleness", "SWC-000", "Chainlink round consumed without a staleness or positivity check", i, 0.9)

        if "unchecked" in low and "{" in line:
            add("unchecked-arithmetic", "SWC-101", "Arithmetic inside an unchecked block can wrap", i, 0.6)

        if "block.timestamp" in line and ("keccak256" in line or "prevrandao" in line or "blockhash" in line):
            add("weak-randomness", "SWC-120", "Randomness derived from block fields the builder controls", i, 0.9)

        if "delegatecall" in low:
            add("delegatecall", "SWC-112", "delegatecall executes foreign code against local storage", i, 0.7)

        if re.search(r"\bfor\s*\(", line) and re.search(r"\.length\b", line):
            add("unbounded-loop", "SWC-128", "Loop bounded by an array length any account can grow", i, 0.7)

        m = _EXTERNAL_CALL.search(line)
        if m:
            if m.group(1) in {"transfer", "transferFrom"} and not re.search(r"(require|bool|=\s*\w+\.)", line):
                add("unchecked-return", "SWC-104", "ERC20 transfer return value is discarded", i, 0.65)
            if call_line is None:
                call_line = i

        if call_line is not None and i > call_line and _STATE_WRITE.search(line):
            var = _STATE_WRITE.search(line).group(1)
            if var not in {"uint256", "uint", "address", "bool", "bytes32", "require"} and " " not in var:
                add("cei-violation", "SWC-107", f"State write to '{var}' happens after an external call (reentrancy window)", i, 0.85)
                call_line = None

    return _dedupe(hits)


def _dedupe(hits: Iterable[PatternHit]) -> list[PatternHit]:
    seen: set[tuple[str, int]] = set()
    out = []
    for h in hits:
        key = (h.rule_id, h.line)
        if key not in seen:
            seen.add(key)
            out.append(h)
    return out


def summarise(spans: list[CodeSpan]) -> dict:
    findings = [f for s in spans for f in s.findings]
    by_rule: dict[str, int] = {}
    for f in findings:
        by_rule[f.rule_id] = by_rule.get(f.rule_id, 0) + 1
    return {
        "spans": len(spans),
        "callables": sum(1 for s in spans if s.kind != "contract"),
        "pattern_hits": len(findings),
        "by_rule": dict(sorted(by_rule.items())),
        "swc_set": sorted({f.swc for f in findings}),
    }
