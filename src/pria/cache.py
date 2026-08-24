"""Semantic cache for the query path.

An exact-key cache is close to useless here because two analysts never phrase
the same question identically. This one keys on the query embedding and serves a
hit when cosine similarity clears a threshold, which is what holds p50 latency
down on a repeated workload without pinning it to exact string equality.

The threshold is deliberately high (0.93 by default): a false hit returns the
wrong evidence, which is far worse than a miss.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    hit_similarities: list[float] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": round(self.hit_rate, 4),
            "mean_hit_similarity": round(float(np.mean(self.hit_similarities)), 4) if self.hit_similarities else None,
        }


class SemanticCache:
    def __init__(self, threshold: float = 0.93, capacity: int = 512, ttl_seconds: float | None = None):
        self.threshold = threshold
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self.stats = CacheStats()
        self._entries: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._matrix: np.ndarray | None = None
        self._keys: list[str] = []

    def _rebuild(self) -> None:
        if not self._entries:
            self._matrix, self._keys = None, []
            return
        self._keys = list(self._entries)
        self._matrix = np.vstack([self._entries[k]["vector"] for k in self._keys])

    def _expired(self, entry: dict[str, Any]) -> bool:
        return self.ttl_seconds is not None and (time.time() - entry["stored_at"]) > self.ttl_seconds

    def get(self, query: str, vector: np.ndarray) -> tuple[Any, float] | None:
        if self._matrix is None:
            self.stats.misses += 1
            return None
        sims = self._matrix @ np.asarray(vector, dtype=np.float32)
        best = int(np.argmax(sims))
        score = float(sims[best])
        key = self._keys[best]
        entry = self._entries.get(key)
        if entry is None or self._expired(entry) or score < self.threshold:
            self.stats.misses += 1
            return None
        self._entries.move_to_end(key)
        self.stats.hits += 1
        self.stats.hit_similarities.append(score)
        return entry["value"], score

    def put(self, query: str, vector: np.ndarray, value: Any) -> None:
        self._entries[query] = {
            "vector": np.asarray(vector, dtype=np.float32),
            "value": value,
            "stored_at": time.time(),
        }
        self._entries.move_to_end(query)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)
            self.stats.evictions += 1
        self._rebuild()

    def wrap(self, query: str, vector: np.ndarray, compute: Callable[[], Any]) -> tuple[Any, bool]:
        hit = self.get(query, vector)
        if hit is not None:
            return hit[0], True
        value = compute()
        self.put(query, vector, value)
        return value, False

    def clear(self) -> None:
        self._entries.clear()
        self._rebuild()

    def __len__(self) -> int:
        return len(self._entries)
