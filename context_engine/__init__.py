"""
Context Awareness Engine for Graph RAG.

Provides intelligent query understanding through:
- Session management for conversation history
- Intent classification (rule-based or LLM)
- Query rewriting with coreference resolution
- Context assembly for retrieval optimization
- Cross-encoder reranking with MMR diversity
- Metadata filtering and recency weighting
- Entity extraction and resolution
- User profile personalization
- Feedback collection and analytics

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

    # Reranking with diversity
    from context_engine import rerank_chunks, mmr_rerank
    reranked = rerank_chunks(query_embedding, chunks, use_mmr=True)

    # Filtering
    from context_engine import FilterCriteria, apply_filters
    criteria = FilterCriteria(max_age_days=90, allowed_sources=["policy"])
    filtered = apply_filters(chunks, criteria)

    # Entity extraction
    from context_engine import extract_entities, EntityType
    entities = extract_entities("Contact John Smith at Acme Corp.")

    # Feedback
    from context_engine import get_feedback_collector
    collector = get_feedback_collector()
    collector.thumbs_up(session_id, query, response, chunk_ids)
"""

from .types import Intent, SessionContext, QueryContext, ConversationTurn
from .session import SessionManager, get_session_manager
from .intent import classify_intent, get_intent_hints
from .query_rewriter import rewrite_query
from .engine import ContextEngine, get_context_engine, process_query

# Phase 2: Reranking and Filtering
from .reranker import (
    rerank_chunks, rerank_by_embedding, mmr_rerank,
    compute_diversity_score, RankedChunk
)
from .filters import (
    FilterCriteria, apply_filters, apply_recency_boost,
    compute_recency_weight, build_cypher_filters
)

# Phase 3: Entities and Profiles
from .entities import (
    Entity, EntityType, extract_entities, extract_entity_strings,
    resolve_coreferences, link_entities, expand_query_with_entities,
    group_entities_by_type
)
from .profiles import (
    UserProfile, UserPreferences, ProfileManager, get_profile_context,
    apply_personalization, compute_personalization_boost
)

# Phase 4: Feedback
from .feedback import (
    FeedbackCollector, FeedbackEntry, FeedbackType, RetrievalMetrics,
    get_feedback_collector
)

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

    # Reranking (Phase 2)
    'rerank_chunks',
    'rerank_by_embedding',
    'mmr_rerank',
    'compute_diversity_score',
    'RankedChunk',

    # Filtering (Phase 2)
    'FilterCriteria',
    'apply_filters',
    'apply_recency_boost',
    'compute_recency_weight',
    'build_cypher_filters',

    # Entities (Phase 3)
    'Entity',
    'EntityType',
    'extract_entities',
    'extract_entity_strings',
    'resolve_coreferences',
    'link_entities',
    'expand_query_with_entities',
    'group_entities_by_type',

    # Profiles (Phase 3)
    'UserProfile',
    'UserPreferences',
    'ProfileManager',
    'get_profile_context',
    'apply_personalization',
    'compute_personalization_boost',

    # Feedback (Phase 4)
    'FeedbackCollector',
    'FeedbackEntry',
    'FeedbackType',
    'RetrievalMetrics',
    'get_feedback_collector',
]
