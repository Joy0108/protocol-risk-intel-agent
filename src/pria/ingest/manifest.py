"""SQLite manifest: the record of what was ingested, from where, and when.

Ingestion is resumable because every unit of work is keyed by content hash and
recorded before the next unit starts. Re-running the pipeline over an unchanged
corpus is a no-op that reports zero new documents, which is what makes a corpus
refresh reproducible rather than a fresh build every time.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id        TEXT PRIMARY KEY,
    source        TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    title         TEXT,
    severity      TEXT,
    swc           TEXT,
    tags          TEXT,
    metadata      TEXT,
    ingested_at   TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'active'
);
CREATE INDEX IF NOT EXISTS idx_documents_hash   ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES documents(doc_id),
    ordinal      INTEGER NOT NULL,
    text         TEXT NOT NULL,
    n_words      INTEGER NOT NULL,
    metadata     TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

CREATE TABLE IF NOT EXISTS duplicates (
    doc_id       TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,          -- 'exact' | 'near'
    duplicate_of TEXT NOT NULL,
    similarity   REAL NOT NULL,
    detected_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    config       TEXT,
    stats        TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class DocumentRecord:
    doc_id: str
    source: str
    content_hash: str
    title: str
    severity: str | None = None
    swc: str | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None
    text: str = ""


class Manifest:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Manifest:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- reads -------------------------------------------------------------
    def known_hashes(self) -> dict[str, str]:
        """Content hash -> the *active* document that owns it.

        Only active documents may own a hash. If a rejected duplicate were
        allowed into this map, a resumed run could resolve the original's hash
        to the duplicate and flip which of the two survives, so the surviving
        document would depend on row order rather than on ingestion order.
        """
        rows = self.conn.execute("SELECT content_hash, doc_id FROM documents WHERE status = 'active'").fetchall()
        return {r["content_hash"]: r["doc_id"] for r in rows}

    def classified_duplicates(self) -> dict[str, str]:
        """doc_id -> content hash for documents already rejected as duplicates."""
        rows = self.conn.execute(
            "SELECT d.doc_id, d.content_hash FROM documents d JOIN duplicates u ON u.doc_id = d.doc_id"
        ).fetchall()
        return {r["doc_id"]: r["content_hash"] for r in rows}

    def has_doc(self, doc_id: str) -> bool:
        return self.conn.execute("SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,)).fetchone() is not None

    def documents(self, source: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM documents WHERE status = 'active'"
        args: tuple = ()
        if source:
            sql += " AND source = ?"
            args = (source,)
        return [self._row_to_doc(r) for r in self.conn.execute(sql + " ORDER BY doc_id", args)]

    def chunks(self) -> list[dict[str, Any]]:
        sql = (
            "SELECT c.chunk_id, c.doc_id, c.ordinal, c.text, c.n_words, c.metadata AS chunk_metadata, "
            "       d.source, d.title, d.severity, d.swc, d.tags, d.metadata "
            "FROM chunks c JOIN documents d ON d.doc_id = c.doc_id "
            "WHERE d.status = 'active' ORDER BY c.doc_id, c.ordinal"
        )
        out = []
        for r in self.conn.execute(sql):
            out.append(
                {
                    "chunk_id": r["chunk_id"],
                    "doc_id": r["doc_id"],
                    "ordinal": r["ordinal"],
                    "text": r["text"],
                    "n_words": r["n_words"],
                    "chunk_metadata": json.loads(r["chunk_metadata"] or "{}"),
                    "source": r["source"],
                    "title": r["title"],
                    "severity": r["severity"],
                    "swc": r["swc"],
                    "tags": json.loads(r["tags"] or "[]"),
                    "metadata": json.loads(r["metadata"] or "{}"),
                }
            )
        return out

    def duplicates(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM duplicates ORDER BY doc_id")]

    def stats(self) -> dict[str, Any]:
        cur = self.conn.execute
        by_source = {r["source"]: r["n"] for r in cur("SELECT source, COUNT(*) n FROM documents WHERE status='active' GROUP BY source")}
        by_sev = {r["severity"]: r["n"] for r in cur("SELECT severity, COUNT(*) n FROM documents WHERE status='active' AND severity IS NOT NULL GROUP BY severity")}
        return {
            "documents": cur("SELECT COUNT(*) n FROM documents WHERE status='active'").fetchone()["n"],
            "chunks": cur("SELECT COUNT(*) n FROM chunks").fetchone()["n"],
            "duplicates_exact": cur("SELECT COUNT(*) n FROM duplicates WHERE kind='exact'").fetchone()["n"],
            "duplicates_near": cur("SELECT COUNT(*) n FROM duplicates WHERE kind='near'").fetchone()["n"],
            "by_source": by_source,
            "by_severity": by_sev,
        }

    # -- writes ------------------------------------------------------------
    def upsert_document(self, rec: DocumentRecord) -> None:
        self.conn.execute(
            "INSERT INTO documents (doc_id, source, content_hash, title, severity, swc, tags, metadata, ingested_at, status)"
            " VALUES (?,?,?,?,?,?,?,?,?, 'active')"
            " ON CONFLICT(doc_id) DO UPDATE SET content_hash=excluded.content_hash, title=excluded.title,"
            " severity=excluded.severity, swc=excluded.swc, tags=excluded.tags, metadata=excluded.metadata,"
            " ingested_at=excluded.ingested_at, status='active'",
            (
                rec.doc_id,
                rec.source,
                rec.content_hash,
                rec.title,
                rec.severity,
                rec.swc,
                json.dumps(list(rec.tags)),
                json.dumps(rec.metadata or {}),
                _now(),
            ),
        )

    def replace_chunks(self, doc_id: str, chunks: Iterable[tuple[str, int, str, dict[str, Any] | None]]) -> int:
        self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        rows = [
            (cid, doc_id, ordinal, text, len(text.split()), json.dumps(meta or {}, default=str))
            for cid, ordinal, text, meta in chunks
        ]
        self.conn.executemany(
            "INSERT INTO chunks (chunk_id, doc_id, ordinal, text, n_words, metadata) VALUES (?,?,?,?,?,?)", rows
        )
        return len(rows)

    def record_duplicate(self, doc_id: str, kind: str, duplicate_of: str, similarity: float) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO duplicates (doc_id, kind, duplicate_of, similarity, detected_at) VALUES (?,?,?,?,?)",
            (doc_id, kind, duplicate_of, float(similarity), _now()),
        )
        self.conn.execute("UPDATE documents SET status = 'duplicate' WHERE doc_id = ?", (doc_id,))

    def start_run(self, run_id: str, config: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO runs (run_id, started_at, config) VALUES (?,?,?)",
            (run_id, _now(), json.dumps(config, default=str)),
        )
        self.conn.commit()

    def finish_run(self, run_id: str, stats: dict[str, Any]) -> None:
        self.conn.execute(
            "UPDATE runs SET finished_at = ?, stats = ? WHERE run_id = ?",
            (_now(), json.dumps(stats, default=str), run_id),
        )
        self.conn.commit()

    def commit(self) -> None:
        self.conn.commit()

    @staticmethod
    def _row_to_doc(r: sqlite3.Row) -> dict[str, Any]:
        d = dict(r)
        d["tags"] = json.loads(d.get("tags") or "[]")
        d["metadata"] = json.loads(d.get("metadata") or "{}")
        return d
