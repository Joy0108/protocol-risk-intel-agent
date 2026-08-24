from __future__ import annotations

from pria.config import RAW_DIR
from pria.ingest.manifest import Manifest
from pria.ingest.minhash import LSHIndex, MinHasher, content_hash, shingles
from pria.ingest.pipeline import run_ingest


def test_content_hash_ignores_whitespace_and_case():
    assert content_hash("The  Vault\nis drained.") == content_hash("the vault is drained")
    assert content_hash("a") != content_hash("b")


def test_minhash_estimates_jaccard_within_tolerance():
    hasher = MinHasher(num_perms=256, shingle_size=3)
    a = "the withdraw function transfers the underlying token before decrementing the share balance of the caller"
    b = "the withdraw function transfers the underlying asset before decrementing the share balance of the caller"
    sa, sb = shingles(a, 3), shingles(b, 3)
    true = len(sa & sb) / len(sa | sb)
    estimate = hasher.signature(a).jaccard(hasher.signature(b))
    assert abs(estimate - true) < 0.12


def test_lsh_surfaces_a_near_duplicate_and_not_an_unrelated_document():
    hasher = MinHasher(num_perms=128, shingle_size=3)
    index = LSHIndex(num_perms=128, bands=32)
    base = "reentrancy in the withdraw path lets the caller re-enter before the balance is decremented"
    index.add("base", hasher.signature(base))
    near = "reentrancy in the withdraw path lets an attacker re-enter before the balance is decremented"
    far = "the governance timelock forwards queued calldata using delegatecall to an arbitrary target"
    assert [k for k, _ in index.query(hasher.signature(near), 0.5)] == ["base"]
    assert index.query(hasher.signature(far), 0.5) == []


def test_ingest_is_resumable_and_idempotent(tmp_path):
    path = tmp_path / "manifest.sqlite"
    first = run_ingest(raw_dir=RAW_DIR, manifest_path=path, resume=False)
    assert first["new"] > 0
    with Manifest(path) as m:
        first_dupes = {d["doc_id"]: d["duplicate_of"] for d in m.duplicates()}
        first_active = {d["doc_id"] for d in m.documents()}

    second = run_ingest(raw_dir=RAW_DIR, manifest_path=path, resume=True)
    assert second["new"] == 0, "a second pass over an unchanged corpus must ingest nothing"
    assert second["skipped_resume"] == second["seen"], "every document should be recognised, not re-decided"
    assert second["manifest"]["chunks"] == first["manifest"]["chunks"]

    with Manifest(path) as m:
        assert {d["doc_id"]: d["duplicate_of"] for d in m.duplicates()} == first_dupes, (
            "which document survives a duplicate pair must not depend on run order"
        )
        assert {d["doc_id"] for d in m.documents()} == first_active


def test_duplicates_are_detected_and_excluded_from_retrieval(manifest_path):
    with Manifest(manifest_path) as manifest:
        dupes = {d["doc_id"]: d for d in manifest.duplicates()}
        active = {d["doc_id"] for d in manifest.documents()}
        chunk_docs = {c["doc_id"] for c in manifest.chunks()}

    assert "c4r-0017-mirror" in dupes and dupes["c4r-0017-mirror"]["kind"] == "exact"
    assert any(d["kind"] == "near" for d in dupes.values())
    for doc_id in dupes:
        assert doc_id not in active, "a duplicate must not stay active"
        assert doc_id not in chunk_docs, "a duplicate must not contribute chunks to the index"


def test_code_chunks_are_declaration_aligned(manifest_path):
    with Manifest(manifest_path) as manifest:
        code = [c for c in manifest.chunks() if c["source"] == "solidity"]
    assert code, "the Solidity fixtures should produce chunks"
    ids = {c["chunk_id"] for c in code}
    assert "sol-VulnerableVault.sol::VulnerableVault.withdraw" in ids
    for chunk in code:
        span = chunk["chunk_metadata"].get("span")
        assert span, "every code chunk carries its span metadata"
        assert span["end_line"] >= span["start_line"]
