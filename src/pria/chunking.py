"""Chunking. Prose gets a sliding word window; Solidity gets declaration spans."""

from __future__ import annotations

from collections.abc import Iterator

from .solidity.ast_lite import extract_spans


def window_chunks(text: str, size: int = 120, overlap: int = 24) -> Iterator[tuple[int, str]]:
    """Sliding word window. Short documents yield exactly one chunk."""
    words = text.split()
    if not words:
        return
    if len(words) <= size:
        yield 0, " ".join(words)
        return
    step = max(1, size - overlap)
    ordinal = 0
    for start in range(0, len(words), step):
        piece = words[start : start + size]
        if not piece:
            break
        yield ordinal, " ".join(piece)
        ordinal += 1
        if start + size >= len(words):
            break


def code_chunks(source: str, path: str) -> Iterator[tuple[int, str, dict]]:
    """One chunk per callable, with the enclosing contract carried as context.

    The signature and the detected pattern hits are prepended to the indexed
    text so a lexical query such as ``ecrecover zero address`` matches the span
    even when the body itself never spells the phrase out.
    """
    spans = extract_spans(source, path)
    callables = [s for s in spans if s.kind != "contract"]
    if not callables:
        callables = spans
    for ordinal, span in enumerate(callables):
        header = f"{span.contract}.{span.name} ({span.kind}) lines {span.start_line}-{span.end_line}"
        hints = "; ".join(f"{f.swc} {f.title}" for f in span.findings)
        body = f"{header}\n{span.signature}\n{span.text}"
        if hints:
            body += f"\nstatic pattern hits: {hints}"
        yield ordinal, body, {
            "span": span.to_dict(),
            "qualified_name": span.qualified_name,
            "start_line": span.start_line,
            "end_line": span.end_line,
        }


def page_blocks(text: str, target_words: int = 26) -> list[str]:
    """Split a PDF page into pseudo visual blocks for late-interaction indexing.

    A real ColQwen2 pass produces one vector per image patch. With text-extracted
    pages the analogue is a vector per layout block, so the MaxSim aggregation
    below has the same shape as the multivector original.
    """
    words = text.split()
    if not words:
        return []
    blocks = []
    for start in range(0, len(words), target_words):
        piece = words[start : start + target_words]
        if piece:
            blocks.append(" ".join(piece))
    return blocks
