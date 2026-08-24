"""MinHash + banded LSH for near-duplicate detection.

The corpus mixes contests that review the same code, so the same root cause is
often reported twice in near-identical prose. Content hashing catches only exact
repeats; this catches the paraphrases.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass

_MERSENNE = (1 << 61) - 1
_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def shingles(text: str, k: int = 5) -> set[str]:
    toks = _tokens(text)
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1)}


def _hash(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")


def _coeffs(num_perms: int) -> list[tuple[int, int]]:
    """Deterministic (a, b) pairs for the universal hash family."""
    out = []
    for i in range(num_perms):
        a = _hash(f"a{i}") % (_MERSENNE - 1) + 1
        b = _hash(f"b{i}") % _MERSENNE
        out.append((a, b))
    return out


@dataclass(frozen=True)
class MinHashSignature:
    values: tuple[int, ...]

    def jaccard(self, other: MinHashSignature) -> float:
        if not self.values:
            return 0.0
        same = sum(1 for x, y in zip(self.values, other.values, strict=True) if x == y)
        return same / len(self.values)


class MinHasher:
    def __init__(self, num_perms: int = 128, shingle_size: int = 5):
        self.num_perms = num_perms
        self.shingle_size = shingle_size
        self._coeffs = _coeffs(num_perms)

    def signature(self, text: str) -> MinHashSignature:
        grams = shingles(text, self.shingle_size)
        if not grams:
            return MinHashSignature(tuple([_MERSENNE] * self.num_perms))
        hashed = [_hash(g) % _MERSENNE for g in grams]
        values = []
        for a, b in self._coeffs:
            values.append(min(((a * h + b) % _MERSENNE) for h in hashed))
        return MinHashSignature(tuple(values))


class LSHIndex:
    """Banded LSH over MinHash signatures.

    ``bands`` trades recall against candidate volume: the probability that two
    documents with Jaccard s become candidates is 1 - (1 - s**r)**b for r rows
    per band.
    """

    def __init__(self, num_perms: int = 128, bands: int = 32):
        if num_perms % bands:
            raise ValueError("num_perms must be divisible by bands")
        self.num_perms = num_perms
        self.bands = bands
        self.rows = num_perms // bands
        self._buckets: dict[tuple[int, int], list[str]] = defaultdict(list)
        self._sigs: dict[str, MinHashSignature] = {}

    def add(self, key: str, sig: MinHashSignature) -> None:
        self._sigs[key] = sig
        for band in range(self.bands):
            chunk = sig.values[band * self.rows : (band + 1) * self.rows]
            self._buckets[(band, _hash(",".join(map(str, chunk))))].append(key)

    def query(self, sig: MinHashSignature, threshold: float = 0.8) -> list[tuple[str, float]]:
        seen: set[str] = set()
        for band in range(self.bands):
            chunk = sig.values[band * self.rows : (band + 1) * self.rows]
            seen.update(self._buckets.get((band, _hash(",".join(map(str, chunk)))), []))
        scored = [(key, sig.jaccard(self._sigs[key])) for key in seen]
        return sorted([(k, s) for k, s in scored if s >= threshold], key=lambda kv: -kv[1])

    def __len__(self) -> int:
        return len(self._sigs)


def content_hash(text: str) -> str:
    """Stable content hash used for exact-duplicate rejection and resumability."""
    normalised = " ".join(_tokens(text))
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()
