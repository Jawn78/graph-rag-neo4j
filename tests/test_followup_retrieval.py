"""Tests for follow-up aware retrieval helpers in qa.answer."""

from dataclasses import dataclass, field
from typing import List, Optional

from graph_rag.qa.answer import (
    _apply_heading_boost,
    _blend_followup_embedding,
    ANCHOR_DOC_BOOST,
    FOLLOWUP_EMB_WEIGHT,
)
from graph_rag.context_engine.types import SessionContext


@dataclass
class FakeCtx:
    is_followup: bool = False
    session: Optional[SessionContext] = None
    anchor_doc_ids: List[str] = field(default_factory=list)


def chunk(cid, doc_id="doc", heading=""):
    return {"chunk_id": cid, "doc_id": doc_id, "heading": heading,
            "text": "t", "order": 0, "title": "T"}


class TestAnchorBoost:
    def test_anchor_doc_promoted(self):
        other = chunk("c1", doc_id="other")
        anchored = chunk("c2", doc_id="cited")
        ordered = _apply_heading_boost("unrelated words", [other, anchored],
                                       anchor_doc_ids=["cited"])
        assert ordered[0]["chunk_id"] == "c2"

    def test_no_anchors_keeps_order(self):
        c1, c2 = chunk("c1"), chunk("c2")
        assert _apply_heading_boost("unrelated words", [c1, c2]) == [c1, c2]

    def test_heading_match_can_outrank_anchor(self):
        # A strong heading match (cap 0.45) beats the anchor boost alone
        anchored = chunk("a", doc_id="cited")
        heading_hit = chunk("h", heading="Dental Coverage Details Overview")
        ordered = _apply_heading_boost(
            "dental coverage details overview question",
            [anchored, heading_hit],
            anchor_doc_ids=["cited"],
        )
        assert ANCHOR_DOC_BOOST < 0.45
        assert ordered[0]["chunk_id"] == "h"


class TestEmbeddingBlend:
    def test_not_followup_unchanged(self):
        ctx = FakeCtx(is_followup=False)
        assert _blend_followup_embedding([1.0, 0.0], ctx) == [1.0, 0.0]

    def test_followup_without_prev_unchanged(self):
        ctx = FakeCtx(is_followup=True, session=SessionContext(session_id="s"))
        assert _blend_followup_embedding([1.0, 0.0], ctx) == [1.0, 0.0]

    def test_followup_blends(self):
        s = SessionContext(session_id="s")
        s.metadata["last_q_emb"] = [0.0, 1.0]
        ctx = FakeCtx(is_followup=True, session=s)
        blended = _blend_followup_embedding([1.0, 0.0], ctx)
        w = FOLLOWUP_EMB_WEIGHT
        assert blended == [w * 1.0, (1 - w) * 1.0]

    def test_dimension_mismatch_unchanged(self):
        s = SessionContext(session_id="s")
        s.metadata["last_q_emb"] = [0.0, 1.0, 0.5]
        ctx = FakeCtx(is_followup=True, session=s)
        assert _blend_followup_embedding([1.0, 0.0], ctx) == [1.0, 0.0]
