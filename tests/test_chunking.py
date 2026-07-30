"""Tests for the ingestion chunker and heading assignment."""

from graph_rag.ingest.files import _chunk_text, _build_chunks, _looks_like_heading


class TestChunkText:
    def test_short_text_single_chunk(self):
        assert _chunk_text("A short sentence.", target_chars=100, overlap=10) == ["A short sentence."]

    def test_respects_target_size(self):
        sentences = " ".join(f"Sentence number {i} is here." for i in range(50))
        chunks = _chunk_text(sentences, target_chars=200, overlap=20)
        assert len(chunks) > 1
        # Each chunk stays near the target (target + overlap slack)
        assert all(len(c) <= 200 + 20 + 1 for c in chunks)

    def test_long_sentence_hard_split(self):
        # A single "sentence" with no punctuation must not produce one giant chunk
        run_on = "word " * 500  # ~2500 chars, no sentence boundary
        chunks = _chunk_text(run_on, target_chars=300, overlap=30)
        assert len(chunks) > 1
        assert all(len(c) <= 300 + 30 + 1 for c in chunks)

    def test_overlap_present(self):
        sentences = ". ".join(f"Sentence {i} has content" for i in range(30)) + "."
        chunks = _chunk_text(sentences, target_chars=150, overlap=25)
        # Every chunk after the first starts with the tail of the previous one
        for i in range(1, len(chunks)):
            assert chunks[i - 1][-10:] in chunks[i][:60]


class TestBuildChunks:
    def _doc(self, doc_id, text):
        return {"doc_id": doc_id, "title": doc_id, "text": text, "metadata": {}}

    def test_heading_detected_and_inherited(self):
        text = "OVERVIEW SECTION\nSome intro. " + "More detail follows here. " * 5
        chunks = _build_chunks([self._doc("d1", text)])
        assert chunks
        assert all(c["heading"] == "OVERVIEW SECTION" for c in chunks)

    def test_heading_does_not_bleed_across_documents(self):
        # Doc 1 has a heading; doc 2 has none. Doc 2's chunks must NOT
        # inherit doc 1's heading (regression test for cross-doc bleed).
        doc1 = self._doc("d1", "BENEFITS OVERVIEW\nDental care is covered for members.")
        doc2 = self._doc("d2", "plain lowercase text with no heading at all in it.")
        chunks = _build_chunks([doc1, doc2])
        d2_chunks = [c for c in chunks if c["doc_id"] == "d2"]
        assert d2_chunks
        assert all(c["heading"] == "" for c in d2_chunks)

    def test_chunk_ids_unique_and_ordered(self):
        text = ". ".join(f"Sentence {i} has plenty of content in it" for i in range(200)) + "."
        chunks = _build_chunks([self._doc("d1", text)])
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))
        assert [c["order"] for c in chunks] == list(range(len(chunks)))


class TestLooksLikeHeading:
    def test_all_caps(self):
        assert _looks_like_heading("EMERGENCY PROCEDURES")

    def test_numbered(self):
        assert _looks_like_heading("11.5 Dental Care")

    def test_markdown(self):
        assert _looks_like_heading("## Setup Guide")

    def test_plain_sentence(self):
        assert not _looks_like_heading("this is just a normal sentence")
