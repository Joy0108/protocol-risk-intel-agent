"""Central configuration. Every tunable that an ablation touches lives here."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("PRIA_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
GOLDEN_PATH = DATA_DIR / "golden" / "golden_set.json"
ARTIFACT_DIR = Path(os.environ.get("PRIA_ARTIFACT_DIR", ROOT / "artifacts"))
REPORT_DIR = Path(os.environ.get("PRIA_REPORT_DIR", ROOT / "reports"))
MANIFEST_PATH = ARTIFACT_DIR / "manifest.sqlite"


@dataclass(frozen=True)
class RetrievalConfig:
    """One row of the ablation matrix.

    Defaults are the configuration that won the ablation; every other row is
    produced by :func:`dataclasses.replace` on this object.
    """

    name: str = "final"

    # --- lexical -----------------------------------------------------------
    use_bm25: bool = True
    bm25_k1: float = 1.2
    bm25_b: float = 0.75

    # --- dense -------------------------------------------------------------
    use_dense: bool = True
    embedder: str = "lsa"  # "lsa" | "hash" | "sentence-transformers"
    st_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 128
    query_prefix: str = "Represent this sentence for searching relevant passages: "

    # --- fusion ------------------------------------------------------------
    fusion: str = "rrf"  # "rrf" | "linear" | "none"
    rrf_k: int = 60
    linear_alpha: float = 0.5  # weight on dense when fusion == "linear"

    # --- filtering and reranking ------------------------------------------
    metadata_filter: bool = True
    # Off by default: see the ablation note in index/rerank.py. Turn on for a
    # corpus large enough that the fused top-40 is still noisy.
    rerank: bool = False
    rerank_backend: str = "feature"  # "feature" | "cross-encoder"
    ce_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_depth: int = 40

    # --- query expansion ---------------------------------------------------
    use_hyde: bool = False  # removed after ablation: costs 0.061 nDCG@10

    # --- serving -----------------------------------------------------------
    quantize: bool = True  # int8 scalar quantization on the dense store
    semantic_cache: bool = True
    cache_threshold: float = 0.93

    top_k: int = 10
    candidate_k: int = 60

    def variant(self, name: str, **overrides) -> RetrievalConfig:
        return replace(self, name=name, **overrides)


@dataclass(frozen=True)
class AgentConfig:
    max_critic_loops: int = 2
    min_citation_rate: float = 0.90
    require_citations: bool = True
    llm_backend: str = os.environ.get("PRIA_LLM", "deterministic")  # or "anthropic"
    anthropic_model: str = "claude-sonnet-5"


@dataclass(frozen=True)
class IngestConfig:
    chunk_words: int = 120
    chunk_overlap_words: int = 24
    minhash_perms: int = 128
    minhash_bands: int = 32
    shingle_size: int = 5
    # 0.50, not the 0.8 used for web-scale dedup: two contests reporting the
    # same root cause reuse the structure but paraphrase the prose, and the
    # hand-labelled duplicate pairs in this corpus sit between 0.53 and 1.0.
    near_dup_threshold: float = 0.50
    sources: tuple[str, ...] = field(
        default_factory=lambda: ("code4rena", "sherlock", "rekt.news", "spearbit", "solidity", "taxonomy")
    )


DEFAULT_RETRIEVAL = RetrievalConfig()
DEFAULT_AGENT = AgentConfig()
DEFAULT_INGEST = IngestConfig()


def ensure_dirs() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
