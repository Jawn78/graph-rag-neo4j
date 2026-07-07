"""
Context Awareness Engine - Main orchestrator.

Combines session management, intent classification, and query rewriting
to produce enriched context for the RAG pipeline.
"""

import logging
import os
import threading
from typing import Any, Dict, List, Optional

from .types import SessionContext, QueryContext
from .session import get_session_manager, SessionManager
from .intent import classify_intent, get_intent_hints
from .query_rewriter import (
    rewrite_query, detect_followup, detect_reformulation, _content_tokens
)

logger = logging.getLogger(__name__)

# Below this token overlap with the previous user query, the conversation is
# considered to have moved to a new topic and the topic stack is cleared.
TOPIC_SHIFT_MIN_OVERLAP = float(os.getenv("TOPIC_SHIFT_MIN_OVERLAP", "0.15"))


class ContextEngine:
    """
    Main context awareness engine.

    Orchestrates the query understanding pipeline:
    1. Session management (conversation history)
    2. Follow-up / topic-shift / reformulation detection
    3. Intent classification
    4. Query rewriting (coreference resolution, expansion)
    5. Context assembly for retrieval
    """

    def __init__(self, session_manager: Optional[SessionManager] = None,
                 use_llm_intent: bool = False,
                 use_llm_rewrite: bool = False):
        """
        Initialize the context engine.

        Args:
            session_manager: Custom session manager (uses global if None)
            use_llm_intent: Use LLM for intent classification
            use_llm_rewrite: Use LLM for query rewriting (follow-ups only;
                self-contained queries always take the fast rule-based path)
        """
        self._session_manager = session_manager or get_session_manager()
        self._use_llm_intent = use_llm_intent
        self._use_llm_rewrite = use_llm_rewrite

    def process_query(self, query: str, session_id: Optional[str] = None) -> QueryContext:
        """
        Process a user query and produce enriched context.

        Args:
            query: The user's raw query
            session_id: Optional session ID for conversation tracking

        Returns:
            QueryContext with all enrichments applied
        """
        # Get or create session
        session = None
        if session_id:
            session = self._session_manager.get_or_create(session_id)

        # Detect follow-up before the new turn is recorded
        is_followup = detect_followup(query, session)

        # Reformulation = implicit negative feedback on the previous answer
        if session:
            prev_query = detect_reformulation(query, session)
            if prev_query:
                self._log_reformulation(session_id, prev_query, query)

        # Topic shift: clear stale topics so they can't hijack coreference
        # resolution ("it" three subjects later must not mean the old topic).
        if session and not is_followup:
            self._maybe_clear_topics(query, session)

        # Classify intent
        intent, confidence = classify_intent(
            query,
            session=session,
            use_llm=self._use_llm_intent
        )
        logger.info(f"Intent: {intent.value} (confidence: {confidence:.2f})")

        # Rewrite query (rewrite_query reserves the LLM for follow-ups)
        rewritten, entities = rewrite_query(
            query,
            session=session,
            use_llm=self._use_llm_rewrite
        )

        if rewritten != query:
            logger.info(f"Query rewritten: '{query[:50]}' -> '{rewritten[:50]}'")

        # Get retrieval hints based on intent
        hints = get_intent_hints(intent)

        # Anchor follow-ups to the documents cited in the previous answer
        anchor_doc_ids: List[str] = []
        if session and is_followup:
            anchor_doc_ids = list(session.metadata.get("last_doc_ids", []))[:5]

        # Build query context
        ctx = QueryContext(
            original_query=query,
            rewritten_query=rewritten,
            intent=intent,
            entities=entities,
            session=session,
            is_followup=is_followup,
            anchor_doc_ids=anchor_doc_ids,
            require_recency=hints["require_recency"],
            require_comparison=hints["require_comparison"],
            require_procedure=hints["require_procedure"],
            diversity_weight=hints["diversity_weight"],
            top_k=int(6 * hints["top_k_multiplier"]),
        )

        # Update session with this turn
        if session_id and session:
            session.add_turn("user", query, intent=intent, entities=entities)

            # Update topic stack
            if entities:
                session.push_topic(entities[0])

            self._session_manager.save(session)

        return ctx

    def add_response(self, session_id: str, response: str,
                     used_chunks: Optional[List[Dict[str, Any]]] = None,
                     query_embedding: Optional[List[float]] = None) -> None:
        """
        Record an assistant response in the session.

        Call this after generating a response to maintain conversation history.

        Args:
            session_id: Session to record into
            response: The generated answer text
            used_chunks: Chunks used to build the answer; their doc_ids become
                anchors for follow-up retrieval
            query_embedding: Embedding of the query that produced this answer;
                blended into the next follow-up's embedding
        """
        if not session_id:
            return

        session = self._session_manager.get_or_create(session_id)
        session.add_turn("assistant", response)

        if used_chunks:
            # Preserve retrieval order; dedupe doc ids
            doc_ids: List[str] = []
            for c in used_chunks:
                d = c.get("doc_id")
                if d and d not in doc_ids:
                    doc_ids.append(d)
            session.metadata["last_doc_ids"] = doc_ids[:5]
            session.metadata["last_chunk_ids"] = [
                c["chunk_id"] for c in used_chunks[:10] if c.get("chunk_id")
            ]

        if query_embedding:
            session.metadata["last_q_emb"] = list(query_embedding)

        self._session_manager.save(session)

    def get_session(self, session_id: str) -> Optional[SessionContext]:
        """Get session context for a session ID."""
        return self._session_manager.get(session_id)

    def clear_session(self, session_id: str) -> None:
        """Clear a session's history."""
        self._session_manager.delete(session_id)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _maybe_clear_topics(self, query: str, session: SessionContext) -> None:
        """Clear the topic stack when the query moves to a new topic."""
        prev = session.get_last_user_query()
        if not prev or not session.topic_stack:
            return

        cur_toks = _content_tokens(query)
        prev_toks = _content_tokens(prev)
        if not cur_toks or not prev_toks:
            return

        overlap = len(cur_toks & prev_toks) / len(cur_toks | prev_toks)
        if overlap < TOPIC_SHIFT_MIN_OVERLAP:
            logger.debug(f"Topic shift detected (overlap {overlap:.2f}); clearing topic stack")
            session.clear_topics()

    def _log_reformulation(self, session_id: str, prev_query: str, new_query: str) -> None:
        """Log a reformulation as implicit negative feedback. Best-effort."""
        try:
            from .feedback import get_feedback_collector
            get_feedback_collector().reformulation(session_id, prev_query, new_query)
            logger.debug(f"Logged reformulation: '{prev_query[:40]}' -> '{new_query[:40]}'")
        except Exception as e:
            logger.debug(f"Reformulation logging failed: {e}")


# Global engine instances, keyed by configuration.
# (A single latched instance silently ignored use_llm on later calls.)
_engines: Dict[bool, ContextEngine] = {}
_engines_lock = threading.Lock()


def get_context_engine(use_llm: bool = False) -> ContextEngine:
    """
    Get or create the global context engine for the given configuration.

    Engines are cached per use_llm value, so callers that request LLM-backed
    processing get it even if a rule-based engine was created first.
    """
    engine = _engines.get(use_llm)
    if engine is None:
        with _engines_lock:
            engine = _engines.get(use_llm)
            if engine is None:
                engine = ContextEngine(
                    use_llm_intent=use_llm,
                    use_llm_rewrite=use_llm
                )
                _engines[use_llm] = engine
    return engine


def process_query(query: str, session_id: Optional[str] = None) -> QueryContext:
    """
    Convenience function to process a query with the global engine.
    """
    return get_context_engine().process_query(query, session_id)
