"""Tests for RRF fusion, FTS query building, heading boost, and context packing."""

from graph_rag.graph.query import rrf_fuse
from graph_rag.qa.answer import (
    _escape_lucene,
    _fts_query_from_question,
    _q_tokens,
    _apply_heading_boost,
    _build_context_block_fit_diverse,
)


def chunk(cid, **kw):
    base = {"chunk_id": cid, "text": f"text {cid}", "heading": "", "order": 0,
            "doc_id": "doc", "title": "T"}
    base.update(kw)
    return base


class TestRrfFuse:
    def test_empty(self):
        assert rrf_fuse([], [], top_k=5) == []

    def test_overlapping_id_wins(self):
        a, b, c = chunk("a"), chunk("b"), chunk("c")
        fused = rrf_fuse(
            [(a, 0.9), (b, 0.5)],       # vector ranking: a=1, b=2
            [(b, 5.0), (c, 1.0)],       # keyword ranking: b=1, c=2
            top_k=3,
        )
        ids = [x["chunk_id"] for x in fused]
        assert ids == ["b", "a", "c"]  # b appears in both -> highest RRF

    def test_score_attached(self):
        a = chunk("a")
        fused = rrf_fuse([(a, 1.0)], [], top_k=1)
        assert fused[0]["_score"] == 1.0 / 61  # rank 1 with k=60

    def test_top_k_respected(self):
        pairs = [(chunk(f"c{i}"), 1.0 - i * 0.01) for i in range(10)]
        assert len(rrf_fuse(pairs, [], top_k=3)) == 3


class TestLuceneEscaping:
    def test_specials_escaped(self):
        assert _escape_lucene("c++") == r"c\+\+"
        assert _escape_lucene("a/b") == r"a\/b"
        assert _escape_lucene('say "hi"') == r'say \"hi\"'

    def test_fts_query_escapes_tokens(self):
        q = _fts_query_from_question("configure TCP/IP networking")
        assert r"tcp\/ip" in q
        assert " OR " in q

    def test_stop_words_filtered(self):
        toks = _q_tokens("what is the difference between apples and oranges")
        assert "the" not in toks
        assert "difference" in toks


class TestHeadingBoost:
    def test_matching_heading_promoted(self):
        c1 = chunk("c1", heading="Unrelated Section")
        c2 = chunk("c2", heading="Dental Coverage Details")
        ordered = _apply_heading_boost("what dental coverage exists", [c1, c2])
        assert ordered[0]["chunk_id"] == "c2"

    def test_no_tokens_no_reorder(self):
        c1, c2 = chunk("c1"), chunk("c2")
        assert _apply_heading_boost("", [c1, c2]) == [c1, c2]

    def test_stable_when_no_match(self):
        c1 = chunk("c1", heading="Alpha")
        c2 = chunk("c2", heading="Beta")
        ordered = _apply_heading_boost("completely unrelated question", [c1, c2])
        assert [c["chunk_id"] for c in ordered] == ["c1", "c2"]


class TestContextPacking:
    def test_round_robin_across_docs(self):
        chunks = [
            chunk("a1", doc_id="docA", order=0, text="A first"),
            chunk("a2", doc_id="docA", order=1, text="A second"),
            chunk("b1", doc_id="docB", order=0, text="B first"),
        ]
        text, cits = _build_context_block_fit_diverse(chunks)
        assert len(cits) == 3
        # Round-robin: docA then docB before docA's second chunk
        assert cits[0].startswith("[docA")
        assert cits[1].startswith("[docB")
        assert cits[2].startswith("[docA")
        assert "A first" in text and "B first" in text

    def test_empty_input(self):
        text, cits = _build_context_block_fit_diverse([])
        assert text == "" and cits == []
