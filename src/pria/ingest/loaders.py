"""Source loaders. Each yields DocumentRecord objects with a normalised shape."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from ..config import RAW_DIR
from .manifest import DocumentRecord
from .minhash import content_hash


def _jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_no} is not valid JSON: {exc}") from exc


def load_contest_findings(raw_dir: Path = RAW_DIR) -> Iterator[DocumentRecord]:
    """Code4rena and Sherlock contest findings."""
    for filename in ("code4rena_findings.jsonl", "sherlock_findings.jsonl"):
        for row in _jsonl(raw_dir / filename):
            text = f"{row['title']}. {row['body']}"
            yield DocumentRecord(
                doc_id=row["id"],
                source=row["source"],
                content_hash=content_hash(text),
                title=row["title"],
                severity=row.get("severity"),
                swc=row.get("swc"),
                tags=tuple(row.get("tags", [])),
                metadata={
                    "contest": row.get("contest"),
                    "date": row.get("date"),
                    "dasp": row.get("dasp"),
                    "kind": "finding",
                },
                text=text,
            )


def load_postmortems(raw_dir: Path = RAW_DIR) -> Iterator[DocumentRecord]:
    """rekt.news incident write-ups."""
    for row in _jsonl(raw_dir / "rekt_postmortems.jsonl"):
        text = f"{row['title']}. {row['body']}"
        yield DocumentRecord(
            doc_id=row["id"],
            source=row["source"],
            content_hash=content_hash(text),
            title=row["title"],
            severity=row.get("severity"),
            swc=row.get("swc"),
            tags=tuple(row.get("tags", [])),
            metadata={
                "incident": row.get("incident"),
                "date": row.get("date"),
                "loss_usd": row.get("loss_usd"),
                "kind": "postmortem",
            },
            text=text,
        )


def load_report_pages(raw_dir: Path = RAW_DIR) -> Iterator[DocumentRecord]:
    """Spearbit / Cantina report pages.

    Pages arrive as extracted text with layout metadata. ``PRIA_PDF_DIR`` can
    point at real PDFs instead; ``pdfplumber`` is then used per page and the rest
    of the pipeline is unchanged.
    """
    for row in _jsonl(raw_dir / "reports" / "spearbit_pages.jsonl"):
        text = f"{row['title']}. {row['text']}"
        yield DocumentRecord(
            doc_id=row["id"],
            source=row["source"],
            content_hash=content_hash(text),
            title=row["title"],
            severity=None,
            swc=None,
            tags=("report", "pdf-page"),
            metadata={
                "report": row.get("report"),
                "page": row.get("page"),
                "layout": row.get("layout"),
                "kind": "report_page",
            },
            text=text,
        )


def load_contracts(raw_dir: Path = RAW_DIR) -> Iterator[DocumentRecord]:
    """Etherscan-verified Solidity sources retained as diagnostic fixtures."""
    contract_dir = raw_dir / "contracts"
    if not contract_dir.exists():
        return
    for path in sorted(contract_dir.glob("*.sol")):
        source = path.read_text(encoding="utf-8")
        yield DocumentRecord(
            doc_id=f"sol-{path.name}",
            source="solidity",
            content_hash=content_hash(source),
            title=path.stem,
            severity=None,
            swc=None,
            tags=("solidity", "source"),
            # as_posix so a manifest built on Windows and one built on Linux
            # record the same path, and so citations stay comparable across them.
            metadata={"path": path.relative_to(raw_dir).as_posix(), "kind": "contract", "lines": source.count("\n") + 1},
            text=source,
        )


def load_taxonomy(raw_dir: Path = RAW_DIR) -> Iterator[DocumentRecord]:
    """SWC and DASP entries, indexed so a bare identifier resolves to its definition."""
    path = raw_dir / "taxonomy.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    for swc_id, entry in data.get("swc", {}).items():
        text = f"{swc_id} {entry['title']}. {entry['summary']} Mapped weakness: {entry.get('cwe', 'n/a')}."
        yield DocumentRecord(
            doc_id=f"tax-{swc_id}",
            source="taxonomy",
            content_hash=content_hash(text),
            title=f"{swc_id}: {entry['title']}",
            severity=None,
            swc=swc_id,
            tags=("taxonomy", "swc"),
            metadata={"cwe": entry.get("cwe"), "kind": "taxonomy"},
            text=text,
        )


ALL_LOADERS = (
    load_contest_findings,
    load_postmortems,
    load_report_pages,
    load_contracts,
    load_taxonomy,
)


def load_all(raw_dir: Path = RAW_DIR) -> Iterator[DocumentRecord]:
    for loader in ALL_LOADERS:
        yield from loader(raw_dir)
