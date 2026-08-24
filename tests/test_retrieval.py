from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from pria.config import DEFAULT_RETRIEVAL
from pria.eval.metrics import ndcg_at_k, reciprocal_rank
from pria.eval.run_eval import evaluate_retrieval
from pria.index.bm25 import BM25Index, tokenize
from pria.index.dense import quantize_int8
from pria.index.hybrid import Retriever, linear_fusion, reciprocal_rank_fusion
from pria.index.multivector import query_vectors


def test_tokenizer_splits_camel_and_snake_case():
    toks = tokenize("latestRoundData and get_virtual_price")
    assert "latestrounddata" in toks and "latest" in toks and "round" in toks
    assert "virtual" in toks and "price" in toks


def test_bm25_ranks_the_rare_term_document_first():
    index = BM25Index().build(
        [
            {"chunk_id": "a", "text": "the vault holds tokens and pays rewards to depositors"},
            {"chunk_id": "b", "text": "ecrecover returns the zero address on a malformed signature"},
            {"chunk_id": "c", "text": "the pool charges a fee on every swap"},
        ]
    )
    assert index.search("ecrecover zero address", top_k=1)[0][0] == "b"


def test_rrf_prefers_a_document_both_retrievers_rank_highly():
    lexical = [("a", 9.0), ("b", 8.0), ("c", 7.0)]
    dense = [("c", 0.9), ("b", 0.8), ("d", 0.7)]
    fused = dict(reciprocal_rank_fusion([lexical, dense], k=10))
    assert fused["b"] > fused["a"], "b is ranked by both, a only by one"
    assert set(fused) == {"a", "b", "c", "d"}


def test_linear_fusion_respects_alpha():
    lexical, dense = [("a", 1.0), ("b", 0.0)], [("b", 1.0), ("a", 0.0)]
    assert linear_fusion(lexical, dense, alpha=0.0)[0][0] == "a"
    assert linear_fusion(lexical, dense, alpha=1.0)[0][0] == "b"


def test_int8_quantization_preserves_ranking_and_shrinks_the_store():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(64, 96)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    store = quantize_int8(vectors)
    recovered = store.decode()
    assert np.abs(recovered - vectors).max() < 0.02
    assert store.nbytes < vectors.nbytes / 3

    q = vectors[0]
    assert int(np.argmax(vectors @ q)) == int(np.argmax(recovered @ q))


def test_metadata_filter_confines_results_to_the_requested_source(retriever):
    payload = retriever.search("bridge validator key compromise", filters={"source": "rekt.news"})
    assert payload["results"]
    assert {r["source"] for r in payload["results"]} == {"rekt.news"}


def test_semantic_cache_serves_a_repeated_query(retriever):
    question = "how does read only reentrancy inflate a curve virtual price"
    retriever.search(question)
    again = retriever.search(question)
    assert again["cache_hit"] is True
    assert again["cache_similarity"] >= DEFAULT_RETRIEVAL.cache_threshold


def test_cache_does_not_serve_across_different_filters(retriever):
    question = "signature replay across chains"
    retriever.search(question, filters=None)
    filtered = retriever.search(question, filters={"source": "sherlock"})
    assert filtered["cache_hit"] is False


def test_query_vectors_cover_the_whole_query():
    windows = query_vectors("chainlink latest round data staleness check", None, window=3)
    assert len(windows) > 1
    joined = " ".join(windows)
    for term in ("chainlink", "staleness", "check"):
        assert term in joined


def test_ndcg_rewards_putting_the_primary_document_first():
    good = ndcg_at_k(["p", "s", "x"], primary=["p"], secondary=["s"], k=10)
    bad = ndcg_at_k(["x", "s", "p"], primary=["p"], secondary=["s"], k=10)
    assert math.isclose(good, 1.0)
    assert bad < good


def test_reciprocal_rank_is_zero_when_nothing_relevant_is_retrieved():
    assert reciprocal_rank(["a", "b"], ["z"]) == 0.0
    assert reciprocal_rank(["a", "z"], ["z"]) == 0.5


@pytest.mark.parametrize("name,overrides", [("bm25", {"use_dense": False}), ("dense", {"use_bm25": False})])
def test_hybrid_is_at_least_as_good_as_either_retriever_alone(chunks, golden, name, overrides):
    single = Retriever(chunks, replace(DEFAULT_RETRIEVAL, fusion="none", **overrides))
    hybrid = Retriever(chunks, DEFAULT_RETRIEVAL)
    solo = evaluate_retrieval(single, golden)[1]["ndcg@10"]
    both = evaluate_retrieval(hybrid, golden)[1]["ndcg@10"]
    assert both >= solo - 1e-9, f"hybrid ({both}) regressed against {name}-only ({solo})"


def test_hyde_expansion_is_a_documented_regression(chunks, golden):
    """The ablation removed HyDE. This test is what stops it coming back."""
    without = evaluate_retrieval(Retriever(chunks, DEFAULT_RETRIEVAL), golden)[1]["ndcg@10"]
    with_hyde = evaluate_retrieval(Retriever(chunks, replace(DEFAULT_RETRIEVAL, use_hyde=True)), golden)[1]["ndcg@10"]
    assert with_hyde < without
