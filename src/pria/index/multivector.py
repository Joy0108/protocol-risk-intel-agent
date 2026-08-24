"""Late-interaction index over report pages (ColQwen2-shaped).

A single vector per PDF page loses the thing that makes review reports useful:
the answer usually lives in one table row or one paragraph, and pooling the
whole page averages it away. Late interaction keeps one vector per layout block
and scores a page as the sum over query vectors of the best-matching block:

    score(q, page) = sum_i max_j  q_i . d_j

which is the MaxSim aggregation ColBERT and ColQwen2 use. ``CaptionIndex`` is
the single-vector baseline the ablation compares against.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..chunking import page_blocks
from .bm25 import tokenize
from .embed import Embedder


def query_vectors(query: str, embedder: Embedder, window: int = 3, stride: int = 1) -> list[str]:
    """Split a query into overlapping term windows, one vector per window."""
    toks = tokenize(query)
    if not toks:
        return [query]
    if len(toks) <= window:
        return [" ".join(toks)]
    return [" ".join(toks[i : i + window]) for i in range(0, len(toks) - window + 1, stride)]


@dataclass
class PageEntry:
    page_id: str
    doc_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    n_blocks: int = 0


class MultiVectorIndex:
    def __init__(self, embedder: Embedder, block_words: int = 26, query_window: int = 3):
        self.embedder = embedder
        self.block_words = block_words
        self.query_window = query_window
        self.pages: list[PageEntry] = []
        self.block_matrix: np.ndarray | None = None  # (total_blocks, dim)
        self.page_slices: list[tuple[int, int]] = []

    def build(self, records: Sequence[dict[str, Any]], text_key: str = "text", id_key: str = "doc_id") -> MultiVectorIndex:
        all_blocks: list[str] = []
        for rec in records:
            blocks = page_blocks(rec[text_key], self.block_words) or [rec[text_key]]
            start = len(all_blocks)
            all_blocks.extend(blocks)
            self.page_slices.append((start, len(all_blocks)))
            self.pages.append(
                PageEntry(page_id=rec[id_key], doc_id=rec[id_key], metadata=rec.get("metadata", {}), n_blocks=len(blocks))
            )
        if not all_blocks:
            self.block_matrix = np.zeros((0, 1), dtype=np.float32)
            return self
        self.embedder.fit(all_blocks)
        self.block_matrix = np.asarray(self.embedder.encode(all_blocks), dtype=np.float32)
        return self

    def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        if self.block_matrix is None or not len(self.pages):
            return []
        windows = query_vectors(query, self.embedder, self.query_window)
        qmat = np.asarray(self.embedder.encode(windows, is_query=True), dtype=np.float32)  # (nq, dim)
        sims = qmat @ self.block_matrix.T  # (nq, n_blocks)

        scored = []
        for entry, (start, end) in zip(self.pages, self.page_slices, strict=True):
            if end <= start:
                scored.append((entry.page_id, 0.0))
                continue
            maxsim = sims[:, start:end].max(axis=1)  # best block per query vector
            scored.append((entry.page_id, float(maxsim.sum() / len(windows))))
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return [(pid, round(s, 6)) for pid, s in scored[:top_k]]

    def stats(self) -> dict[str, Any]:
        return {
            "pages": len(self.pages),
            "blocks": 0 if self.block_matrix is None else int(self.block_matrix.shape[0]),
            "blocks_per_page": round(np.mean([p.n_blocks for p in self.pages]), 2) if self.pages else 0.0,
            "dim": 0 if self.block_matrix is None else int(self.block_matrix.shape[1]),
        }


class CaptionIndex:
    """Single-vector-per-page baseline standing in for a VLM caption pass.

    A caption model produces one short description of the page; the analogue
    here is the page title plus its opening sentence, embedded once. This is the
    configuration the late-interaction index is measured against.
    """

    def __init__(self, embedder: Embedder, caption_words: int = 40):
        self.embedder = embedder
        self.caption_words = caption_words
        self.ids: list[str] = []
        self.matrix: np.ndarray | None = None

    def build(
        self, records: Sequence[dict[str, Any]], text_key: str = "text", id_key: str = "doc_id", refit: bool = True
    ) -> CaptionIndex:
        captions = [" ".join(r[text_key].split()[: self.caption_words]) for r in records]
        self.ids = [r[id_key] for r in records]
        if not captions:
            self.matrix = np.zeros((0, 1), dtype=np.float32)
            return self
        if refit:
            self.embedder.fit(captions)
        self.matrix = np.asarray(self.embedder.encode(captions), dtype=np.float32)
        return self

    def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        if self.matrix is None or not len(self.ids):
            return []
        q = np.asarray(self.embedder.encode([query], is_query=True), dtype=np.float32)[0]
        sims = self.matrix @ q
        order = np.argsort(-sims)[:top_k]
        return [(self.ids[i], round(float(sims[i]), 6)) for i in order]
