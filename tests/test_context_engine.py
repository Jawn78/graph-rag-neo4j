"""Tests for context-awareness: follow-ups, rewriting, topics, intent, entities."""

from graph_rag.context_engine.types import SessionContext, Intent
from graph_rag.context_engine.query_rewriter import (
    detect_followup,
    detect_reformulation,
    _parse_rewrite_response,
    _content_tokens,
)
from graph_rag.context_engine.intent import classify_intent_rules
from graph_rag.context_engine.corpus_entities import CorpusEntityDictionary
from graph_rag.context_engine.engine import ContextEngine
from graph_rag.context_engine.session import SessionManager, InMemorySessionStore


def make_session(*user_queries):
    s = SessionContext(session_id="t1")
    for q in user_queries:
        s.add_turn("user", q, entities=[])
    return s


class TestDetectFollowup:
    def test_no_session_never_followup(self):
        assert not detect_followup("what about it", None)

    def test_empty_session_never_followup(self):
        assert not detect_followup("what about it", SessionContext(session_id="x"))

    def test_coreference_is_followup(self):
        s = make_session("What does the dental plan cover?")
        assert detect_followup("does it cover implants", s)

    def test_short_query_without_entities_is_followup(self):
        s = make_session("Tell me about parental leave policy")
        assert detect_followup("how many weeks", s)

    def test_selfcontained_query_is_not_followup(self):
        s = make_session("What does the dental plan cover?")
        assert not detect_followup("What is the parental leave policy duration?", s)


class TestDetectReformulation:
    def test_high_overlap_is_reformulation(self):
        s = make_session("what does the dental plan cover for implants")
        prev = detect_reformulation("what dental plan coverage exists for implants", s)
        assert prev == "what does the dental plan cover for implants"

    def test_different_query_is_not(self):
        s = make_session("what does the dental plan cover")
        assert detect_reformulation("how do I file a travel expense report", s) is None

    def test_identical_query_is_not(self):
        s = make_session("same question")
        assert detect_reformulation("same question", s) is None


class TestParseRewriteResponse:
    def test_valid_json(self):
        raw = '{"query": "does the dental plan cover implants", "entities": ["dental plan"]}'
        q, e = _parse_rewrite_response(raw, "does it cover implants")
        assert q == "does the dental plan cover implants"
        assert e == ["dental plan"]

    def test_json_in_code_fence(self):
        raw = '```json\n{"query": "rewritten", "entities": []}\n```'
        q, _ = _parse_rewrite_response(raw, "orig")
        assert q == "rewritten"

    def test_garbage_falls_back_to_original(self):
        q, e = _parse_rewrite_response("Sure! Here is my answer about dogs...", "orig query")
        assert q == "orig query"
        assert e == []

    def test_runaway_rewrite_rejected(self):
        raw = '{"query": "' + "blah " * 200 + '", "entities": []}'
        q, _ = _parse_rewrite_response(raw, "short")
        assert q == "short"

    def test_empty_query_field_falls_back(self):
        q, _ = _parse_rewrite_response('{"query": "", "entities": ["x"]}', "orig")
        assert q == "orig"


class TestIntentRules:
    def test_comparison_beats_factual_shape(self):
        intent, conf = classify_intent_rules("What is the difference between HMO and PPO?")
        assert intent == Intent.COMPARISON
        assert conf >= 0.8

    def test_how_to(self):
        intent, _ = classify_intent_rules("How do I submit an expense report?")
        assert intent == Intent.HOW_TO

    def test_plain_question_factual(self):
        intent, _ = classify_intent_rules("When does open enrollment start?")
        assert intent == Intent.FACTUAL

    def test_empty(self):
        assert classify_intent_rules("")[0] == Intent.UNKNOWN


class TestCorpusEntities:
    def _dict(self):
        d = CorpusEntityDictionary(ttl_seconds=9999)
        d.load_entries(["HIPAA Privacy Rule", "Dental Care", "benefits_overview_2024"])
        return d

    def test_lowercase_query_matches(self):
        assert self._dict().match("what does the hipaa privacy rule say") == ["HIPAA Privacy Rule"]

    def test_filename_style_alias(self):
        assert self._dict().match("summarize benefits overview 2024") == ["benefits_overview_2024"]

    def test_no_midword_match(self):
        # "dental care" must not match inside "accidental careless"
        assert self._dict().match("accidental careless mistake") == []

    def test_no_match(self):
        assert self._dict().match("unrelated question entirely") == []


class TestEngineFollowupFlow:
    def _engine(self):
        return ContextEngine(session_manager=SessionManager(store=InMemorySessionStore()))

    def test_anchor_docs_flow_from_response_to_followup(self):
        engine = self._engine()
        engine.process_query("What does the dental plan cover?", session_id="s1")
        engine.add_response(
            "s1", "It covers cleanings. [doc]",
            used_chunks=[{"chunk_id": "c1", "doc_id": "docA"},
                         {"chunk_id": "c2", "doc_id": "docB"}],
            query_embedding=[0.1, 0.2],
        )
        ctx = engine.process_query("does it cover implants", session_id="s1")
        assert ctx.is_followup
        assert ctx.anchor_doc_ids == ["docA", "docB"]
        assert ctx.session.metadata["last_q_emb"] == [0.1, 0.2]

    def test_new_topic_has_no_anchors(self):
        engine = self._engine()
        engine.process_query("What does the dental plan cover?", session_id="s2")
        engine.add_response("s2", "answer", used_chunks=[{"chunk_id": "c", "doc_id": "d"}])
        ctx = engine.process_query(
            "What is the corporate travel reimbursement procedure for international flights?",
            session_id="s2",
        )
        assert not ctx.is_followup
        assert ctx.anchor_doc_ids == []

    def test_topic_stack_cleared_on_shift(self):
        engine = self._engine()
        ctx1 = engine.process_query("Tell me about the Dental Plan coverage", session_id="s3")
        session = ctx1.session
        assert session.topic_stack  # entity pushed
        engine.process_query(
            "What is the corporate travel reimbursement procedure for international flights?",
            session_id="s3",
        )
        # Old dental topic must not survive an unrelated query
        assert "Dental Plan" not in session.topic_stack


class TestContentTokens:
    def test_basic(self):
        assert _content_tokens("What does the dental plan cover?") >= {"dental", "plan", "cover"}
