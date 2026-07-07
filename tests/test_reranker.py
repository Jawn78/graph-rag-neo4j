"""Tests for embedding rerank, MMR diversity, and batch embedding resolution."""

from graph_rag.context_engine.reranker import (
    _cosine_similarity,
    _prefetch_embeddings,
    rerank_by_embedding,
    mmr_rerank,
    compute_diversity_score,
)


def chunk(cid, emb=None):
    return {"chunk_id": cid, "text": cid, "embedding": emb}


class TestCosine:
    def test_identical(self):
        assert abs(_cosine_similarity([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-9

    def test_orthogonal(self):
        assert abs(_cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-9

    def test_mismatched_dims(self):
        assert _cosine_similarity([1.0], [1.0, 0.0]) == 0.0

    def test_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


class TestPrefetch:
    def test_uses_inline_embeddings_without_driver(self):
        chunks = [chunk("a", [1.0]), chunk("b")]
        result = _prefetch_embeddings(chunks, driver=None)
        assert result == {"a": [1.0]}


class TestRerankByEmbedding:
    def test_orders_by_similarity(self):
        q = [1.0, 0.0]
        far = chunk("far", [0.0, 1.0])
        near = chunk("near", [1.0, 0.1])
        result = rerank_by_embedding(q, [far, near], top_k=2)
        assert [c["chunk_id"] for c in result] == ["near", "far"]

    def test_fallback_to_score(self):
        q = [1.0, 0.0]
        high = dict(chunk("high"), _score=0.9)
        low = dict(chunk("low"), _score=0.1)
        result = rerank_by_embedding(q, [low, high], top_k=2)
        assert result[0]["chunk_id"] == "high"


class TestMmr:
    def test_prefers_diversity_over_near_duplicate(self):
        q = [1.0, 0.0]
        a = chunk("a", [0.9, 0.44])         # most relevant
        dup = chunk("dup", [0.9, 0.45])     # near-duplicate of a
        div = chunk("div", [0.88, -0.44])   # similarly relevant but diverse
        result = mmr_rerank(q, [a, dup, div], lambda_param=0.5, top_k=2)
        assert result[0]["chunk_id"] == "a"
        assert result[1]["chunk_id"] == "div"

    def test_single_chunk_passthrough(self):
        c = chunk("only", [1.0])
        assert mmr_rerank([1.0], [c]) == [c]


class TestDiversityScore:
    def test_identical_chunks_low_diversity(self):
        chunks = [chunk("a", [1.0, 0.0]), chunk("b", [1.0, 0.0])]
        assert compute_diversity_score(chunks) < 0.01

    def test_orthogonal_chunks_high_diversity(self):
        chunks = [chunk("a", [1.0, 0.0]), chunk("b", [0.0, 1.0])]
        assert compute_diversity_score(chunks) > 0.99

    def test_single_chunk(self):
        assert compute_diversity_score([chunk("a", [1.0])]) == 1.0
