"""
Context Awareness Engine - Main orchestrator.

Combines session management, intent classification, and query rewriting
to produce enriched context for the RAG pipeline.
"""

import logging
from typing import Optional

from .types import Intent, SessionContext, QueryContext
from .session import get_session_manager, SessionManager
from .intent import classify_intent, get_intent_hints
from .query_rewriter import rewrite_query

logger = logging.getLogger(__name__)


class ContextEngine:
    """
    Main context awareness engine.

    Orchestrates the query understanding pipeline:
    1. Session management (conversation history)
    2. Intent classification
    3. Query rewriting (coreference resolution, expansion)
    4. Context assembly for retrieval
    """

    def __init__(self, session_manager: Optional[SessionManager] = None,
                 use_llm_intent: bool = False,
                 use_llm_rewrite: bool = False):
        """
        Initialize the context engine.

        Args:
            session_manager: Custom session manager (uses global if None)
            use_llm_intent: Use LLM for intent classification
            use_llm_rewrite: Use LLM for query rewriting
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

        # Classify intent
        intent, confidence = classify_intent(
            query,
            session=session,
            use_llm=self._use_llm_intent
        )
        logger.info(f"Intent: {intent.value} (confidence: {confidence:.2f})")

        # Rewrite query
        rewritten, entities = rewrite_query(
            query,
            session=session,
            use_llm=self._use_llm_rewrite
        )

        if rewritten != query:
            logger.info(f"Query rewritten: '{query[:50]}' -> '{rewritten[:50]}'")

        # Get retrieval hints based on intent
        hints = get_intent_hints(intent)

        # Build query context
        ctx = QueryContext(
            original_query=query,
            rewritten_query=rewritten,
            intent=intent,
            entities=entities,
            session=session,
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

    def add_response(self, session_id: str, response: str) -> None:
        """
        Record an assistant response in the session.

        Call this after generating a response to maintain conversation history.
        """
        if session_id:
            self._session_manager.add_assistant_turn(session_id, response)

    def get_session(self, session_id: str) -> Optional[SessionContext]:
        """Get session context for a session ID."""
        return self._session_manager.get(session_id)

    def clear_session(self, session_id: str) -> None:
        """Clear a session's history."""
        self._session_manager.delete(session_id)


# Global engine instance
_engine: Optional[ContextEngine] = None


def get_context_engine(use_llm: bool = False) -> ContextEngine:
    """
    Get or create the global context engine.

    Args:
        use_llm: Enable LLM-based intent and rewriting (first call only)
    """
    global _engine
    if _engine is None:
        _engine = ContextEngine(
            use_llm_intent=use_llm,
            use_llm_rewrite=use_llm
        )
    return _engine


def process_query(query: str, session_id: Optional[str] = None) -> QueryContext:
    """
    Convenience function to process a query with the global engine.
    """
    return get_context_engine().process_query(query, session_id)
