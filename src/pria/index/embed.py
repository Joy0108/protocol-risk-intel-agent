"""Pluggable embedding backends.

The query path has no GPU, so the default backend is a corpus-fitted LSA
projection: a TF-IDF matrix reduced with a truncated SVD. It generalises over
synonymy (which BM25 cannot) at a few milliseconds per query on CPU.

``sentence-transformers`` with ``BAAI/bge-small-en-v1.5`` is the production
backend and is selected with ``embedder="sentence-transformers"``. It is an
optional dependency: nothing in the default install or in CI imports torch.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Protocol

import numpy as np

from .bm25 import tokenize


class Embedder(Protocol):
    dim: int

    def fit(self, corpus: Sequence[str]) -> Embedder: ...
    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray: ...


def _l2(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class LSAEmbedder:
    """TF-IDF + truncated SVD. Deterministic, CPU-only, no model download."""

    def __init__(self, dim: int = 128, min_df: int = 1, max_df_ratio: float = 0.9):
        self.dim = dim
        self.min_df = min_df
        self.max_df_ratio = max_df_ratio
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.components: np.ndarray | None = None  # (n_terms, dim)

    def fit(self, corpus: Sequence[str]) -> LSAEmbedder:
        tokenised = [tokenize(t) for t in corpus]
        df = Counter()
        for toks in tokenised:
            df.update(set(toks))
        n = max(1, len(corpus))
        keep = [t for t, d in df.items() if d >= self.min_df and d / n <= self.max_df_ratio]
        self.vocab = {t: i for i, t in enumerate(sorted(keep))}
        if not self.vocab:
            self.idf = np.zeros(0)
            self.components = np.zeros((0, self.dim))
            return self

        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1.0 for t in sorted(keep)], dtype=np.float64)
        matrix = np.zeros((len(corpus), len(self.vocab)), dtype=np.float64)
        for row, toks in enumerate(tokenised):
            counts = Counter(toks)
            for term, count in counts.items():
                col = self.vocab.get(term)
                if col is not None:
                    matrix[row, col] = 1.0 + math.log(count)
        matrix *= self.idf
        matrix = _l2(matrix)

        k = int(min(self.dim, min(matrix.shape) - 1)) or 1
        _, _, vt = np.linalg.svd(matrix, full_matrices=False)
        self.components = vt[:k].T  # (n_terms, k)
        self.dim = k
        return self

    def _tfidf(self, texts: Sequence[str]) -> np.ndarray:
        assert self.idf is not None
        out = np.zeros((len(texts), len(self.vocab)), dtype=np.float64)
        for row, text in enumerate(texts):
            counts = Counter(tokenize(text))
            for term, count in counts.items():
                col = self.vocab.get(term)
                if col is not None:
                    out[row, col] = 1.0 + math.log(count)
        out *= self.idf
        return _l2(out)

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        if self.components is None:
            raise RuntimeError("LSAEmbedder.fit must be called before encode")
        if not len(self.vocab):
            return np.zeros((len(texts), max(1, self.dim)))
        return _l2(self._tfidf(texts) @ self.components)


class HashEmbedder:
    """Feature-hashed TF-IDF-free baseline. Used as an ablation floor."""

    def __init__(self, dim: int = 512, ngram: int = 1):
        self.dim = dim
        self.ngram = ngram

    def fit(self, corpus: Sequence[str]) -> HashEmbedder:
        return self

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim))
        for row, text in enumerate(texts):
            toks = tokenize(text)
            grams = toks + [" ".join(toks[i : i + 2]) for i in range(len(toks) - 1)] if self.ngram > 1 else toks
            for gram in grams:
                h = hash_token(gram)
                out[row, h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        return _l2(out)


class SentenceTransformerEmbedder:  # pragma: no cover - optional dependency
    """bge-small-en-v1.5 through sentence-transformers.

    bge expects an instruction prefix on the query side only; encoding a query
    without it costs several points of nDCG, so the prefix is applied here
    rather than left to the caller.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", query_prefix: str = ""):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.query_prefix = query_prefix
        self.dim = self.model.get_sentence_embedding_dimension()

    def fit(self, corpus: Sequence[str]) -> SentenceTransformerEmbedder:
        return self

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        payload = [self.query_prefix + t for t in texts] if (is_query and self.query_prefix) else list(texts)
        return np.asarray(self.model.encode(payload, normalize_embeddings=True, show_progress_bar=False))


def hash_token(token: str) -> int:
    h = 1469598103934665603
    for byte in token.encode("utf-8"):
        h ^= byte
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def build_embedder(kind: str, dim: int, model_name: str = "", query_prefix: str = "") -> Embedder:
    if kind == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name or "BAAI/bge-small-en-v1.5", query_prefix)
    if kind == "hash":
        return HashEmbedder(dim=dim)
    if kind == "lsa":
        return LSAEmbedder(dim=dim)
    raise ValueError(f"unknown embedder backend: {kind!r}")


_WS = re.compile(r"\s+")


def normalise_text(text: str) -> str:
    return _WS.sub(" ", text).strip()


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _l2(np.atleast_2d(a)) @ _l2(np.atleast_2d(b)).T


def batched(items: Iterable, size: int):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
