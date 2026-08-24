"""Resumable ingestion: load, dedupe, chunk, record.

Ordering matters. Exact content hashes are checked first because they are free,
then MinHash LSH catches the paraphrases, and only surviving documents are
chunked. Every decision is written to the manifest before the next document is
touched, so an interrupted run resumes without redoing work or double counting.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..chunking import code_chunks, window_chunks
from ..config import DEFAULT_INGEST, MANIFEST_PATH, RAW_DIR, IngestConfig, ensure_dirs
from ..tracing import current
from .loaders import load_all
from .manifest import Manifest
from .minhash import LSHIndex, MinHasher


def run_ingest(
    raw_dir: Path = RAW_DIR,
    manifest_path: Path = MANIFEST_PATH,
    cfg: IngestConfig = DEFAULT_INGEST,
    resume: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    ensure_dirs()
    run_id = uuid.uuid4().hex[:12]
    tracer = current()

    stats = {
        "run_id": run_id,
        "seen": 0,
        "new": 0,
        "skipped_resume": 0,
        "exact_duplicates": 0,
        "near_duplicates": 0,
        "chunks": 0,
        "code_spans": 0,
    }

    with Manifest(manifest_path) as manifest:
        manifest.start_run(run_id, asdict(cfg))
        known = manifest.known_hashes() if resume else {}
        already_rejected = manifest.classified_duplicates() if resume else {}

        hasher = MinHasher(num_perms=cfg.minhash_perms, shingle_size=cfg.shingle_size)
        lsh = LSHIndex(num_perms=cfg.minhash_perms, bands=cfg.minhash_bands)
        # Seed the LSH index with what is already active so a resumed run can
        # still detect a near-duplicate of a document ingested last time.
        if resume:
            bodies: dict[str, list[str]] = {}
            for chunk in manifest.chunks():
                bodies.setdefault(chunk["doc_id"], []).append(chunk["text"])
            for doc_id, parts in bodies.items():
                lsh.add(doc_id, hasher.signature(" ".join(parts)))

        with tracer.span("ingest", run_id=run_id) as span:
            for record in load_all(raw_dir):
                stats["seen"] += 1

                if already_rejected.get(record.doc_id) == record.content_hash:
                    # Already classified as a duplicate on a previous run and
                    # unchanged since. Re-deciding it would be wasted work and
                    # could reach a different verdict.
                    stats["skipped_resume"] += 1
                    continue

                prior = known.get(record.content_hash)
                if prior == record.doc_id:
                    stats["skipped_resume"] += 1
                    continue
                if prior is not None:
                    manifest.upsert_document(record)
                    manifest.record_duplicate(record.doc_id, "exact", prior, 1.0)
                    stats["exact_duplicates"] += 1
                    manifest.commit()
                    continue

                sig = hasher.signature(record.text)
                near = lsh.query(sig, threshold=cfg.near_dup_threshold)
                if near:
                    other, score = near[0]
                    manifest.upsert_document(record)
                    manifest.record_duplicate(record.doc_id, "near", other, score)
                    stats["near_duplicates"] += 1
                    manifest.commit()
                    if verbose:
                        print(f"  near-duplicate {record.doc_id} ~ {other} (jaccard {score:.2f})")
                    continue

                manifest.upsert_document(record)
                n_chunks = _chunk_document(manifest, record, cfg, stats)
                lsh.add(record.doc_id, sig)
                known[record.content_hash] = record.doc_id
                stats["new"] += 1
                stats["chunks"] += n_chunks
                manifest.commit()

            span["attributes"].update(stats)

        manifest.finish_run(run_id, stats)
        stats["manifest"] = manifest.stats()

    if verbose:
        print(
            f"ingest {run_id}: {stats['new']} new, {stats['exact_duplicates']} exact dupes, "
            f"{stats['near_duplicates']} near dupes, {stats['chunks']} chunks"
        )
    return stats


def _chunk_document(manifest: Manifest, record, cfg: IngestConfig, stats: dict) -> int:
    if record.source == "solidity":
        rows = []
        for ordinal, text, meta in code_chunks(record.text, record.metadata.get("path", record.doc_id)):
            rows.append((f"{record.doc_id}::{meta['qualified_name']}", ordinal, text, meta))
            stats["code_spans"] += 1
        if not rows:
            rows = [(f"{record.doc_id}::0", 0, record.text, {})]
        return manifest.replace_chunks(record.doc_id, rows)

    rows = [
        (f"{record.doc_id}::{ordinal}", ordinal, text, {})
        for ordinal, text in window_chunks(record.text, cfg.chunk_words, cfg.chunk_overlap_words)
    ]
    return manifest.replace_chunks(record.doc_id, rows)
