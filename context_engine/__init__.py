"""
Context Awareness Engine for Graph RAG.

Provides intelligent query understanding through:
- Session management for conversation history
- Intent classification (rule-based or LLM)
- Query rewriting with coreference resolution
- Context assembly for retrieval optimization

Usage:
    from context_engine import process_query, get_context_engine

    # Simple usage
    ctx = process_query("What does it cover?", session_id="user123")
    print(ctx.rewritten_query)  # Query with resolved references
    print(ctx.intent)           # Classified intent

    # With engine instance
    engine = get_context_engine(use_llm=True)
    ctx = engine.process_query(query, session_id)
    # ... do retrieval and generation ...
    engine.add_response(session_id, answer)
"""

from .types import Intent, SessionContext, QueryContext, ConversationTurn
from .session import SessionManager, get_session_manager
from .intent import classify_intent, get_intent_hints
from .query_rewriter import rewrite_query
from .engine import ContextEngine, get_context_engine, process_query

__all__ = [
    # Types
    'Intent',
    'SessionContext',
    'QueryContext',
    'ConversationTurn',

    # Session management
    'SessionManager',
    'get_session_manager',

    # Intent classification
    'classify_intent',
    'get_intent_hints',

    # Query rewriting
    'rewrite_query',

    # Main engine
    'ContextEngine',
    'get_context_engine',
    'process_query',
]
