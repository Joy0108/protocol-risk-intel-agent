from __future__ import annotations

import pytest

from pria.config import DEFAULT_RETRIEVAL, MANIFEST_PATH, RAW_DIR
from pria.eval.run_eval import load_chunks, load_golden
from pria.index.hybrid import Retriever
from pria.ingest.pipeline import run_ingest


@pytest.fixture(scope="session")
def manifest_path(tmp_path_factory):
    """Ingest the real corpus into a throwaway manifest once per session."""
    path = tmp_path_factory.mktemp("artifacts") / "manifest.sqlite"
    run_ingest(raw_dir=RAW_DIR, manifest_path=path, resume=False)
    return path


@pytest.fixture(scope="session")
def chunks(manifest_path):
    return load_chunks(manifest_path)


@pytest.fixture(scope="session")
def retriever(chunks):
    return Retriever(chunks, DEFAULT_RETRIEVAL)


@pytest.fixture(scope="session")
def golden():
    return load_golden()


@pytest.fixture(scope="session")
def committed_chunks():
    """The chunks from the committed manifest, when one exists."""
    if not MANIFEST_PATH.exists():
        pytest.skip("no committed manifest; run `make ingest` first")
    return load_chunks(MANIFEST_PATH)
