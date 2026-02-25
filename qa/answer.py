"""
Question answering pipeline with hybrid retrieval.

Implements:
- Hybrid search (vector + keyword) with RRF fusion
- Heading-based boost for relevance re-ranking
- Context packing with token budget management
- MCP fallback chain with graceful degradation

All magic numbers are documented with their rationale.
"""

import os
import math
import re
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable

from ..config import (
    embed_client, chat_client, EMBED_MODEL, CHAT_MODEL, NEO4J_DB,
    mcp_client, MCP_CHAT_MODEL, MCP_AGENT_URL
)
from .mcp_adapter import call_mcp_agent
from ..graph.query import hybrid_search
from ..utils.text import sanitize_answer
from ..utils.logging import get_trace_id, set_trace_id, timed

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION - All magic numbers documented
# ============================================================================

# System prompt for answer generation
ANSWER_SYSTEM = (
    "You are a precise assistant. Use ONLY the provided context to answer.\n"
    "- Be concise and factual.\n"
    "- If the answer is not in the context, say you don't know.\n"
    "- Cite supporting chunk(s) inline like [doc:title#chunk_order]."
)

# Token estimation: ~4 characters per token is typical for English text
# This is used for rough context budget calculations
AVG_CHARS_PER_TOKEN = float(os.getenv("AVG_CHARS_PER_TOKEN", "4"))

# Context token budget: Llama-2-7b has 4096 context window
# Reserve ~1000 tokens for system prompt + output
# 2800 leaves comfortable margin for context
MAX_CTX_TOKENS = int(os.getenv("MAX_CTX_TOKENS", "2800"))

# Output token limit: 192 tokens ~= 150 words
# Sufficient for concise, factual answers
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "192"))

# Per-chunk character limit: 900 chars ~= 225 tokens
# Prevents any single chunk from dominating context
PER_CHUNK_CHAR_CAP = int(os.getenv("PER_CHUNK_CHAR_CAP", "900"))

# Neighbor window: ±2 chunks around best FTS match
# Provides local context without overwhelming with off-topic content
FOCUS_WINDOW = int(os.getenv("FOCUS_WINDOW", "2"))

# Max chunks per document: 3 chunks prevents single-doc bias
# Ensures diverse sources in answers
PER_DOC_MAX = int(os.getenv("PER_DOC_MAX", "3"))

# Round-robin chunk selection: interleave chunks from different documents
# This improves answer diversity
DOC_ROUND_ROBIN = os.getenv("DOC_ROUND_ROBIN", "1") == "1"

# Stop words for heading matching
STOP = {
    "the", "a", "an", "of", "in", "to", "and", "or", "for", "over", "under",
    "within", "on", "by", "with", "is", "are", "was", "were", "be", "been",
    "as", "that", "this", "these", "those", "at", "from", "it", "its",
    "their", "his", "her", "about", "into", "out", "up", "down", "not",
    "no", "do", "does", "did"
}
STOP_EXTRA = {
    t.strip().lower() for t in os.getenv(
        "STOP_EXTRA", "air,force,department,united,states,us,u.s.,guardian,space"
    ).split(",") if t.strip()
}

# Heading boost parameters for re-ranking
# Minimum token length to consider for heading matching
HEADING_TOKEN_MINLEN = int(os.getenv("HEADING_TOKEN_MINLEN", "5"))

# Boost per matching token in heading: 0.12 gives moderate weight
HEADING_UNIT_BOOST = float(os.getenv("HEADING_UNIT_BOOST", "0.12"))

# Maximum total heading boost: 0.45 caps heading influence
HEADING_BOOST_CAP = float(os.getenv("HEADING_BOOST_CAP", "0.45"))

# Bonus for bigram phrase matches: 0.30 rewards phrase matches
HEADING_PHRASE_BONUS = float(os.getenv("HEADING_PHRASE_BONUS", "0.30"))


# ============================================================================
# RESPONSE TYPES
# ============================================================================

@dataclass
class LLMResponse:
    """Wrapper for LLM response to handle different client types."""
    content: str
    source: str  # 'agent', 'mcp', 'chat'

    @property
    def choices(self):
        """Compatibility with OpenAI response format."""
        return [_Choice(self.content)]


@dataclass
class _Choice:
    """Internal choice wrapper."""
    def __init__(self, content: str):
        self.message = _Message(content)


@dataclass
class _Message:
    """Internal message wrapper."""
    content: str


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _approx_tokens(text: str) -> int:
    """Estimate token count from text length."""
    return max(1, math.ceil(len(text) / AVG_CHARS_PER_TOKEN))


def _q_tokens(question: str) -> List[str]:
    """Extract significant tokens from question for heading matching."""
    q = (question or "").lower()
    toks = re.findall(r"[a-z0-9][a-z0-9\-_/]*", q)
    return [
        t for t in toks
        if len(t) >= HEADING_TOKEN_MINLEN
        and t not in STOP
        and t not in STOP_EXTRA
    ]


def _extract_phrases(question: str) -> List[str]:
    """Extract bigram phrases from question tokens."""
    toks = _q_tokens(question)
    phrases = []
    for i in range(len(toks) - 1):
        a, b = toks[i], toks[i + 1]
        if a != b:
            phrases.append(f"{a} {b}")
    # Deduplicate while preserving order
    seen, out = set(), []
    for p in phrases:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out[:5]  # Limit to 5 phrases


def _fts_query_from_question(q: str) -> str:
    """Build full-text search query from question."""
    toks = _q_tokens(q)
    phrases = _extract_phrases(q)
    parts = []
    parts += [f'"{p}"' for p in phrases]  # Quoted phrases
    parts += toks  # Individual tokens
    return " OR ".join(parts) if parts else (q or "")


# ============================================================================
# CONTEXT BUILDING
# ============================================================================

def _build_context_block_fit_diverse(chunks: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """
    Pack chunks into context string with diversity and token budget.

    Returns (context_text, list_of_citations).
    """
    # Group by document
    doc_order, by_doc = [], {}
    for ch in chunks:
        d = ch["doc_id"]
        if d not in by_doc:
            by_doc[d] = []
            doc_order.append(d)
        by_doc[d].append(ch)

    def chunk_iterator():
        """Yield chunks in round-robin or sequential order."""
        if not DOC_ROUND_ROBIN:
            for d in doc_order:
                for ch in by_doc[d][:PER_DOC_MAX]:
                    yield ch
        else:
            lanes = [by_doc[d][:PER_DOC_MAX] for d in doc_order]
            i = 0
            while True:
                emitted = False
                for lane in lanes:
                    if i < len(lane):
                        yield lane[i]
                        emitted = True
                if not emitted:
                    break
                i += 1

    lines, cits, used = [], [], 0
    for ch in chunk_iterator():
        tag = f"[{ch['doc_id'][:8]}:{(ch.get('title') or 'Untitled')}#{ch['order']}]"
        text = (ch.get("text") or "")[:PER_CHUNK_CHAR_CAP]
        block = f"{tag}\n{text}\n\n"
        tks = _approx_tokens(block)
        if used + tks > MAX_CTX_TOKENS:
            break
        lines.append(block)
        cits.append(tag)
        used += tks

    return "".join(lines), cits


# ============================================================================
# SEARCH HELPERS
# ============================================================================

def _fts_best_chunk(driver, query_str: str) -> Optional[Tuple[str, int]]:
    """Find the best FTS match and return (doc_id, order)."""
    if not query_str:
        return None

    from neo4j import RoutingControl
    trace_id = get_trace_id()

    try:
        with driver.session(database=NEO4J_DB, default_access_mode=RoutingControl.READ) as s:
            rows = s.run("""
                CALL db.index.fulltext.queryNodes('chunk_text_fts', $q, {limit: 1})
                YIELD node, score
                RETURN node.doc_id AS doc_id, node.`order` AS ord
            """, q=query_str).data()
        return (rows[0]["doc_id"], rows[0]["ord"]) if rows else None
    except Exception as e:
        logger.warning(f"[{trace_id}] FTS best chunk failed: {e}")
        return None


def _neighbor_window(driver, doc_id: str, center_ord: int, window: int) -> List[Dict[str, Any]]:
    """Get chunks within ±window of center_ord in the same document."""
    from neo4j import RoutingControl
    trace_id = get_trace_id()

    lo, hi = max(0, center_ord - window), center_ord + window

    try:
        with driver.session(database=NEO4J_DB, default_access_mode=RoutingControl.READ) as s:
            rows = s.run("""
                MATCH (d:Document {doc_id:$doc})-[:HAS_CHUNK]->(c:Chunk)
                WHERE c.`order` >= $lo AND c.`order` <= $hi
                RETURN c.chunk_id AS chunk_id, c.text AS text, c.heading AS heading,
                       c.`order` AS `order`, c.doc_id AS doc_id, d.title AS title
                ORDER BY c.`order`
            """, doc=doc_id, lo=lo, hi=hi).data()
    except Exception as e:
        logger.warning(f"[{trace_id}] Neighbor window query failed: {e}")
        return []

    return [
        {
            "chunk_id": r["chunk_id"],
            "text": r["text"] or "",
            "heading": r.get("heading") or "",
            "order": r["order"],
            "doc_id": r["doc_id"],
            "title": r["title"] or ""
        }
        for r in rows
    ]


def _hydrate_headings(driver, chunks: List[Dict[str, Any]]) -> None:
    """Fill in missing headings from database."""
    need = [c for c in chunks if not c.get("heading")]
    if not need:
        return

    from neo4j import RoutingControl
    trace_id = get_trace_id()
    ids = [c["chunk_id"] for c in need]

    try:
        with driver.session(database=NEO4J_DB, default_access_mode=RoutingControl.READ) as s:
            rows = s.run("""
                MATCH (c:Chunk) WHERE c.chunk_id IN $ids
                RETURN c.chunk_id AS chunk_id, c.heading AS heading
            """, ids=ids).data()
    except Exception as e:
        logger.warning(f"[{trace_id}] Heading hydration failed: {e}")
        return

    hmap = {r["chunk_id"]: (r.get("heading") or "") for r in rows}

    for c in chunks:
        if not c.get("heading"):
            h = hmap.get(c["chunk_id"], "")
            if not h:
                # Fall back to first line or sentence
                t = (c.get("text") or "").strip()
                h = t.split("\n", 1)[0].strip()
                if len(h) < 12:
                    m = re.split(r"(?<=[\.\!\?])\s+", t, maxsplit=1)
                    h = (m[0] if m else t)[:120].strip()
            c["heading"] = h


def _apply_heading_boost(question: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Re-rank chunks by heading relevance to question."""
    qwords = set(_q_tokens(question))
    phrases = [p.lower() for p in _extract_phrases(question)]

    if not qwords and not phrases:
        return chunks

    # Ensure all chunks have heading field
    for c in chunks:
        if "heading" not in c or c["heading"] is None:
            c["heading"] = ""

    scored = []
    for idx, c in enumerate(chunks):
        h = (c.get("heading") or "").lower()
        hits = sum(1 for w in qwords if w in h)
        boost = min(HEADING_BOOST_CAP, HEADING_UNIT_BOOST * hits) if hits else 0.0

        if boost < HEADING_BOOST_CAP and any(p in h for p in phrases):
            boost = min(HEADING_BOOST_CAP, boost + HEADING_PHRASE_BONUS)

        scored.append((boost, idx, c))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return [c for _, _, c in scored]


# ============================================================================
# LLM FALLBACK CHAIN
# ============================================================================

def _call_agent_gateway(messages: List[Dict], trace_id: str) -> Optional[LLMResponse]:
    """Try the agent gateway (MCP_AGENT_URL)."""
    if not MCP_AGENT_URL:
        return None

    logger.debug(f"[{trace_id}] Trying agent gateway: {MCP_AGENT_URL}")

    try:
        ok, content_or_err = call_mcp_agent(messages, agent_url=MCP_AGENT_URL)
        if ok:
            return LLMResponse(content=content_or_err, source='agent')
        else:
            logger.warning(f"[{trace_id}] Agent gateway failed: {content_or_err}")
            return None
    except Exception as e:
        logger.warning(f"[{trace_id}] Agent gateway exception: {e}")
        return None


def _call_mcp_client(messages: List[Dict], trace_id: str) -> Optional[LLMResponse]:
    """Try the MCP client directly."""
    logger.debug(f"[{trace_id}] Trying MCP client")

    try:
        resp = mcp_client.chat.completions.create(
            model=MCP_CHAT_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=MAX_OUTPUT_TOKENS
        )
        content = getattr(resp.choices[0].message, "content", None) or ""
        return LLMResponse(content=content, source='mcp')
    except Exception as e:
        logger.warning(f"[{trace_id}] MCP client failed: {e}")
        return None


def _call_chat_client(messages: List[Dict], trace_id: str) -> Optional[LLMResponse]:
    """Try the default chat client."""
    logger.debug(f"[{trace_id}] Trying chat client")

    try:
        resp = chat_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=MAX_OUTPUT_TOKENS
        )
        content = getattr(resp.choices[0].message, "content", None) or ""
        return LLMResponse(content=content, source='chat')
    except Exception as e:
        logger.warning(f"[{trace_id}] Chat client failed: {e}")
        return None


def _get_llm_response(messages: List[Dict], use_mcp: bool, trace_id: str) -> Tuple[Optional[LLMResponse], Optional[str]]:
    """
    Get LLM response using fallback chain.

    Order when use_mcp=True:
        1. Agent gateway (MCP_AGENT_URL)
        2. MCP client
        3. Default chat client

    Order when use_mcp=False:
        1. Default chat client only

    Returns (response, error_message).
    """
    clients: List[Tuple[str, Callable]] = []

    if use_mcp:
        clients.append(('agent', lambda: _call_agent_gateway(messages, trace_id)))
        clients.append(('mcp', lambda: _call_mcp_client(messages, trace_id)))
    clients.append(('chat', lambda: _call_chat_client(messages, trace_id)))

    last_error = None
    for name, caller in clients:
        try:
            result = caller()
            if result is not None:
                logger.info(f"[{trace_id}] Using {name} response")
                return result, None
        except Exception as e:
            last_error = f"{name}: {e}"
            logger.warning(f"[{trace_id}] {name} failed: {e}")

    return None, f"All LLM clients failed. Last error: {last_error}"


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

@timed("ask")
def ask(driver, question: str, top_k: int = 6, use_mcp: bool = False) -> Dict[str, Any]:
    """
    Answer a question using hybrid search and LLM generation.

    Args:
        driver: Neo4j driver
        question: The question to answer
        top_k: Number of chunks to retrieve (default 6)
        use_mcp: Whether to try MCP services (default False)

    Returns:
        Dict with 'answer', 'citations', 'used_chunks', and optionally 'error'
    """
    trace_id = set_trace_id()
    logger.info(f"[{trace_id}] Processing question: {question[:50]}...")

    # 1) Embed the question
    try:
        q_emb = embed_client.embeddings.create(
            model=EMBED_MODEL,
            input=question,
            encoding_format="float"
        ).data[0].embedding
    except Exception as e:
        logger.error(f"[{trace_id}] Failed to embed question: {e}")
        return {"answer": "", "citations": [], "used_chunks": [], "error": f"Embedding failed: {e}"}

    # 2) Build FTS query
    fts_q = _fts_query_from_question(question)
    logger.debug(f"[{trace_id}] FTS query: {fts_q}")

    # 3) Anchor on best FTS hit + neighbor window
    ctx_chunks: List[Dict[str, Any]] = []
    best = _fts_best_chunk(driver, fts_q)
    if best:
        doc_id, ord0 = best
        ctx_chunks = _neighbor_window(driver, doc_id, ord0, FOCUS_WINDOW)
        logger.debug(f"[{trace_id}] Anchor: doc={doc_id[:8]}, order={ord0}, got {len(ctx_chunks)} neighbors")

    # 4) Supplement with hybrid search
    try:
        hyb = hybrid_search(driver, q_emb, fts_q, top_k=top_k) or []
    except Exception as e:
        logger.warning(f"[{trace_id}] Hybrid search failed: {e}")
        hyb = []

    # Merge hybrid results (deduplicate)
    seen = {c["chunk_id"] for c in ctx_chunks}
    for h in hyb:
        if h["chunk_id"] not in seen:
            h.setdefault("heading", None)
            ctx_chunks.append(h)
            seen.add(h["chunk_id"])

    logger.info(f"[{trace_id}] Retrieved {len(ctx_chunks)} chunks")

    # 5) Hydrate headings and apply boost
    _hydrate_headings(driver, ctx_chunks)
    ctx_chunks = _apply_heading_boost(question, ctx_chunks)

    # 6) Pack context and build prompt
    context_text, citations = _build_context_block_fit_diverse(ctx_chunks)
    user_prompt = (
        "Answer the question using ONLY the context below. "
        "Be concise, list key facts, and include inline citations.\n\n"
        f"Question:\n{question}\n\nContext:\n{context_text}"
    ).strip()

    messages = [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": user_prompt}
    ]

    # 7) Get LLM response using fallback chain
    response, error = _get_llm_response(messages, use_mcp, trace_id)

    if error:
        logger.error(f"[{trace_id}] LLM failed: {error}")
        return {"answer": "", "citations": [], "used_chunks": ctx_chunks, "error": error}

    # 8) Clean up answer
    answer = sanitize_answer(response.content) if response else ""

    # Add fallback note if we didn't use MCP when requested
    if use_mcp and response and response.source == 'chat':
        answer += "\n\n[Note: MCP request failed; response from default chat backend.]"

    logger.info(f"[{trace_id}] Generated answer ({len(answer)} chars) with {len(citations)} citations")

    return {
        "answer": answer,
        "citations": citations,
        "used_chunks": ctx_chunks
    }


# ============================================================================
# CONTEXT-AWARE ENTRY POINT
# ============================================================================

@timed("ask_with_context")
def ask_with_context(driver, question: str, session_id: Optional[str] = None,
                     top_k: int = 6, use_mcp: bool = False,
                     use_llm_context: bool = False) -> Dict[str, Any]:
    """
    Answer a question using context-aware retrieval.

    This is an enhanced version of ask() that:
    - Tracks conversation history via session_id
    - Classifies query intent
    - Rewrites queries to resolve coreferences
    - Adjusts retrieval strategy based on intent

    Args:
        driver: Neo4j driver
        question: The question to answer
        session_id: Optional session ID for conversation tracking
        top_k: Base number of chunks to retrieve (may be adjusted by intent)
        use_mcp: Whether to try MCP services
        use_llm_context: Use LLM for intent/rewriting (slower but better)

    Returns:
        Dict with 'answer', 'citations', 'used_chunks', 'intent', 'rewritten_query'
    """
    from ..context_engine import get_context_engine, Intent

    trace_id = set_trace_id()
    logger.info(f"[{trace_id}] Processing with context: {question[:50]}...")

    # Get context engine and process query
    engine = get_context_engine(use_llm=use_llm_context)
    ctx = engine.process_query(question, session_id)

    logger.info(f"[{trace_id}] Intent: {ctx.intent.value}, entities: {ctx.entities[:3]}")

    if ctx.rewritten_query != question:
        logger.info(f"[{trace_id}] Rewritten: {ctx.rewritten_query[:50]}...")

    # Use rewritten query for retrieval
    effective_query = ctx.rewritten_query
    effective_top_k = ctx.top_k

    # 1) Embed the rewritten question
    try:
        q_emb = embed_client.embeddings.create(
            model=EMBED_MODEL,
            input=effective_query,
            encoding_format="float"
        ).data[0].embedding
    except Exception as e:
        logger.error(f"[{trace_id}] Failed to embed question: {e}")
        return {
            "answer": "",
            "citations": [],
            "used_chunks": [],
            "intent": ctx.intent.value,
            "rewritten_query": effective_query,
            "error": f"Embedding failed: {e}"
        }

    # 2) Build FTS query
    fts_q = _fts_query_from_question(effective_query)

    # 3) Anchor on best FTS hit + neighbor window (if enabled)
    ctx_chunks: List[Dict[str, Any]] = []
    if ctx.include_neighbors:
        best = _fts_best_chunk(driver, fts_q)
        if best:
            doc_id, ord0 = best
            ctx_chunks = _neighbor_window(driver, doc_id, ord0, FOCUS_WINDOW)
            logger.debug(f"[{trace_id}] Anchor: doc={doc_id[:8]}, order={ord0}")

    # 4) Supplement with hybrid search
    try:
        hyb = hybrid_search(driver, q_emb, fts_q, top_k=effective_top_k) or []
    except Exception as e:
        logger.warning(f"[{trace_id}] Hybrid search failed: {e}")
        hyb = []

    # Merge results
    seen = {c["chunk_id"] for c in ctx_chunks}
    for h in hyb:
        if h["chunk_id"] not in seen:
            h.setdefault("heading", None)
            ctx_chunks.append(h)
            seen.add(h["chunk_id"])

    logger.info(f"[{trace_id}] Retrieved {len(ctx_chunks)} chunks")

    # 5) Hydrate headings and apply boost
    _hydrate_headings(driver, ctx_chunks)
    ctx_chunks = _apply_heading_boost(effective_query, ctx_chunks)

    # 6) Build intent-aware system prompt
    system_prompt = _get_intent_system_prompt(ctx.intent)

    # 7) Pack context and build prompt
    context_text, citations = _build_context_block_fit_diverse(ctx_chunks)
    user_prompt = _get_intent_user_prompt(ctx.intent, question, context_text)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # 8) Get LLM response
    response, error = _get_llm_response(messages, use_mcp, trace_id)

    if error:
        logger.error(f"[{trace_id}] LLM failed: {error}")
        return {
            "answer": "",
            "citations": [],
            "used_chunks": ctx_chunks,
            "intent": ctx.intent.value,
            "rewritten_query": effective_query,
            "error": error
        }

    # 9) Clean up answer and record in session
    answer = sanitize_answer(response.content) if response else ""

    if use_mcp and response and response.source == 'chat':
        answer += "\n\n[Note: MCP request failed; response from default chat backend.]"

    # Record response in session for future context
    if session_id:
        engine.add_response(session_id, answer)

    logger.info(f"[{trace_id}] Generated answer ({len(answer)} chars)")

    return {
        "answer": answer,
        "citations": citations,
        "used_chunks": ctx_chunks,
        "intent": ctx.intent.value,
        "rewritten_query": effective_query,
        "entities": ctx.entities,
    }


def _get_intent_system_prompt(intent) -> str:
    """Get system prompt tailored to query intent."""
    from ..context_engine import Intent

    base = (
        "You are a precise assistant. Use ONLY the provided context to answer.\n"
        "If the answer is not in the context, say you don't know.\n"
        "Cite supporting chunk(s) inline like [doc:title#chunk_order].\n"
    )

    if intent == Intent.HOW_TO:
        return base + "\nProvide step-by-step instructions when applicable."

    elif intent == Intent.COMPARISON:
        return base + "\nStructure your answer to clearly compare the items."

    elif intent == Intent.SUMMARIZATION:
        return base + "\nProvide a concise summary of the key points."

    elif intent == Intent.LIST:
        return base + "\nFormat your answer as a clear list."

    elif intent == Intent.DEFINITION:
        return base + "\nProvide a clear, concise definition."

    return base + "\nBe concise and factual."


def _get_intent_user_prompt(intent, question: str, context: str) -> str:
    """Get user prompt tailored to query intent."""
    from ..context_engine import Intent

    base = f"Question:\n{question}\n\nContext:\n{context}"

    if intent == Intent.HOW_TO:
        return f"Provide step-by-step instructions for the following question, using the context.\n\n{base}"

    elif intent == Intent.COMPARISON:
        return f"Compare and contrast the following, using the context.\n\n{base}"

    elif intent == Intent.SUMMARIZATION:
        return f"Summarize the following based on the context.\n\n{base}"

    elif intent == Intent.LIST:
        return f"List the items requested, using the context.\n\n{base}"

    return f"Answer the following question using ONLY the context below. Be concise.\n\n{base}"


# ============================================================================
# FULL CONTEXT ENGINE INTEGRATION (All Phases)
# ============================================================================

@timed("ask_full_context")
def ask_full_context(driver, question: str, session_id: Optional[str] = None,
                     user_id: Optional[str] = None, top_k: int = 6,
                     use_mcp: bool = False, use_llm_context: bool = False,
                     enable_rerank: bool = True, enable_mmr: bool = True,
                     enable_filtering: bool = True, enable_personalization: bool = True,
                     enable_feedback: bool = True) -> Dict[str, Any]:
    """
    Full context-aware question answering with all phases integrated.

    This comprehensive function integrates:
    - Phase 1: Query understanding (intent, rewriting, session)
    - Phase 2: Reranking with MMR diversity, metadata filtering
    - Phase 3: Entity extraction, user profile personalization
    - Phase 4: Feedback logging and analytics

    Args:
        driver: Neo4j driver
        question: The question to answer
        session_id: Optional session ID for conversation tracking
        user_id: Optional user ID for personalization
        top_k: Base number of chunks to retrieve
        use_mcp: Whether to try MCP services
        use_llm_context: Use LLM for intent/rewriting
        enable_rerank: Enable cross-encoder reranking
        enable_mmr: Enable MMR diversity (requires enable_rerank)
        enable_filtering: Enable metadata filtering
        enable_personalization: Enable user profile personalization
        enable_feedback: Enable feedback logging

    Returns:
        Dict with full result including answer, citations, metrics, etc.
    """
    import time
    import uuid

    from ..context_engine import (
        get_context_engine, Intent,
        extract_entities, expand_query_with_entities,
        rerank_chunks, compute_diversity_score,
        FilterCriteria, apply_filters, apply_recency_boost,
        ProfileManager, apply_personalization, get_profile_context,
        get_feedback_collector
    )

    trace_id = set_trace_id()
    query_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info(f"[{trace_id}] Full context processing: {question[:50]}...")

    # Initialize result structure
    result = {
        "answer": "",
        "citations": [],
        "used_chunks": [],
        "intent": None,
        "rewritten_query": question,
        "entities": [],
        "diversity_score": 0.0,
        "query_id": query_id,
        "metrics": {}
    }

    # -------------------------------------------------------------------------
    # Phase 1: Query Understanding
    # -------------------------------------------------------------------------
    engine = get_context_engine(use_llm=use_llm_context)
    ctx = engine.process_query(question, session_id)

    result["intent"] = ctx.intent.value
    result["rewritten_query"] = ctx.rewritten_query
    result["entities"] = ctx.entities

    logger.info(f"[{trace_id}] Intent: {ctx.intent.value}, entities: {ctx.entities[:3]}")

    # Extract entities from query for expansion
    query_entities = extract_entities(question)
    entity_strings = [e.text for e in query_entities]
    result["entities"] = list(set(result["entities"] + entity_strings))

    # Expand query with entity terms
    effective_query = ctx.rewritten_query
    if query_entities:
        effective_query = expand_query_with_entities(effective_query, query_entities)
        logger.debug(f"[{trace_id}] Entity-expanded query: {effective_query[:80]}...")

    effective_top_k = ctx.top_k

    # -------------------------------------------------------------------------
    # Phase 3 (partial): Load user profile if available
    # -------------------------------------------------------------------------
    user_profile = None
    profile_context = ""
    if enable_personalization and user_id:
        try:
            profile_manager = ProfileManager()
            user_profile = profile_manager.get_or_create(user_id)
            profile_context = get_profile_context(user_profile)

            # Record query in profile
            profile_manager.record_query(
                user_id, question, ctx.intent.value,
                topics=ctx.entities[:5],
                entities=entity_strings[:5]
            )
            logger.debug(f"[{trace_id}] Loaded profile for user: {user_id}")
        except Exception as e:
            logger.warning(f"[{trace_id}] Profile loading failed: {e}")

    # -------------------------------------------------------------------------
    # Embedding
    # -------------------------------------------------------------------------
    try:
        q_emb = embed_client.embeddings.create(
            model=EMBED_MODEL,
            input=effective_query,
            encoding_format="float"
        ).data[0].embedding
    except Exception as e:
        logger.error(f"[{trace_id}] Failed to embed question: {e}")
        result["error"] = f"Embedding failed: {e}"
        return result

    # -------------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------------
    retrieval_start = time.time()

    fts_q = _fts_query_from_question(effective_query)
    ctx_chunks: List[Dict[str, Any]] = []

    # Anchor on FTS if enabled
    if ctx.include_neighbors:
        best = _fts_best_chunk(driver, fts_q)
        if best:
            doc_id, ord0 = best
            ctx_chunks = _neighbor_window(driver, doc_id, ord0, FOCUS_WINDOW)

    # Hybrid search - get more candidates for reranking
    rerank_multiplier = 3 if enable_rerank else 1
    try:
        hyb = hybrid_search(driver, q_emb, fts_q, top_k=effective_top_k * rerank_multiplier) or []
    except Exception as e:
        logger.warning(f"[{trace_id}] Hybrid search failed: {e}")
        hyb = []

    # Merge results
    seen = {c["chunk_id"] for c in ctx_chunks}
    for h in hyb:
        if h["chunk_id"] not in seen:
            h.setdefault("heading", None)
            ctx_chunks.append(h)
            seen.add(h["chunk_id"])

    retrieval_time = (time.time() - retrieval_start) * 1000
    logger.info(f"[{trace_id}] Retrieved {len(ctx_chunks)} chunks in {retrieval_time:.1f}ms")

    # -------------------------------------------------------------------------
    # Phase 2: Filtering
    # -------------------------------------------------------------------------
    if enable_filtering and ctx_chunks:
        # Build filter criteria based on intent
        criteria = FilterCriteria()

        # Require recency for clarification questions
        if ctx.intent == Intent.CLARIFICATION:
            criteria.require_recency = True
            criteria.max_age_days = 365

        # Apply user preference filters
        if user_profile:
            if user_profile.preferences.preferred_sources:
                criteria.allowed_sources = user_profile.preferences.preferred_sources
            if user_profile.preferences.blocked_sources:
                criteria.blocked_sources = user_profile.preferences.blocked_sources
            if user_profile.preferences.preferred_doc_types:
                criteria.allowed_extensions = user_profile.preferences.preferred_doc_types

        # Apply filters
        if not criteria.is_empty():
            pre_filter_count = len(ctx_chunks)
            ctx_chunks = apply_filters(ctx_chunks, criteria)
            logger.debug(f"[{trace_id}] Filtered: {pre_filter_count} -> {len(ctx_chunks)}")

        # Apply recency boost
        ctx_chunks = apply_recency_boost(ctx_chunks)

    # -------------------------------------------------------------------------
    # Phase 2: Reranking with MMR
    # -------------------------------------------------------------------------
    rerank_time = 0.0
    if enable_rerank and ctx_chunks and q_emb:
        rerank_start = time.time()

        # Determine lambda based on intent
        if ctx.intent == Intent.COMPARISON:
            mmr_lambda = 0.5  # More diversity for comparisons
        elif ctx.intent == Intent.HOW_TO:
            mmr_lambda = 0.9  # Less diversity, more coherent steps
        else:
            mmr_lambda = 0.7  # Default balance

        ctx_chunks = rerank_chunks(
            q_emb, ctx_chunks, driver,
            use_mmr=enable_mmr,
            lambda_param=mmr_lambda,
            top_k=effective_top_k
        )

        rerank_time = (time.time() - rerank_start) * 1000
        logger.debug(f"[{trace_id}] Reranked to {len(ctx_chunks)} chunks in {rerank_time:.1f}ms")

    # -------------------------------------------------------------------------
    # Phase 3: Personalization
    # -------------------------------------------------------------------------
    if enable_personalization and user_profile and ctx_chunks:
        ctx_chunks = apply_personalization(ctx_chunks, user_profile)
        # Re-sort by boosted score if applicable
        if any("_personalization_boost" in c for c in ctx_chunks):
            ctx_chunks.sort(
                key=lambda c: c.get("_score", 0) * c.get("_personalization_boost", 1.0),
                reverse=True
            )

    # Compute diversity score
    diversity_score = 0.0
    if ctx_chunks:
        diversity_score = compute_diversity_score(ctx_chunks, driver)
        result["diversity_score"] = diversity_score

    # -------------------------------------------------------------------------
    # Heading boost and context building
    # -------------------------------------------------------------------------
    _hydrate_headings(driver, ctx_chunks)
    ctx_chunks = _apply_heading_boost(effective_query, ctx_chunks)

    context_text, citations = _build_context_block_fit_diverse(ctx_chunks)
    result["citations"] = citations
    result["used_chunks"] = ctx_chunks

    # -------------------------------------------------------------------------
    # Build prompt with profile context
    # -------------------------------------------------------------------------
    system_prompt = _get_intent_system_prompt(ctx.intent)
    if profile_context:
        system_prompt = f"{system_prompt}\n\nUser context:\n{profile_context}"

    user_prompt = _get_intent_user_prompt(ctx.intent, question, context_text)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # -------------------------------------------------------------------------
    # LLM Generation
    # -------------------------------------------------------------------------
    response, error = _get_llm_response(messages, use_mcp, trace_id)

    if error:
        logger.error(f"[{trace_id}] LLM failed: {error}")
        result["error"] = error
        return result

    answer = sanitize_answer(response.content) if response else ""

    if use_mcp and response and response.source == 'chat':
        answer += "\n\n[Note: MCP request failed; response from default chat backend.]"

    result["answer"] = answer

    # Record response in session
    if session_id:
        engine.add_response(session_id, answer)

    # -------------------------------------------------------------------------
    # Phase 4: Feedback logging
    # -------------------------------------------------------------------------
    total_time = (time.time() - start_time) * 1000

    result["metrics"] = {
        "retrieval_time_ms": retrieval_time,
        "rerank_time_ms": rerank_time,
        "total_time_ms": total_time,
        "num_chunks_retrieved": len(ctx_chunks),
        "diversity_score": diversity_score,
    }

    if enable_feedback:
        try:
            collector = get_feedback_collector()
            collector.log_retrieval(
                query_id=query_id,
                query=question,
                intent=ctx.intent.value,
                chunks=ctx_chunks,
                retrieval_time_ms=retrieval_time,
                rerank_time_ms=rerank_time,
                diversity_score=diversity_score
            )
            logger.debug(f"[{trace_id}] Logged retrieval metrics")
        except Exception as e:
            logger.warning(f"[{trace_id}] Feedback logging failed: {e}")

    logger.info(f"[{trace_id}] Generated answer ({len(answer)} chars) in {total_time:.1f}ms")

    return result


def submit_feedback(query_id: str, session_id: str, query: str, response: str,
                    feedback_type: str, chunk_ids: List[str] = None,
                    user_id: str = None, score: int = None,
                    reason: str = None) -> Optional[str]:
    """
    Submit user feedback for a query response.

    Args:
        query_id: The query ID from ask_full_context result
        session_id: Session ID
        query: The original query
        response: The generated response
        feedback_type: "thumbs_up", "thumbs_down", or "rating"
        chunk_ids: IDs of chunks used in response
        user_id: Optional user ID
        score: Rating score 1-5 (required for "rating" type)
        reason: Optional reason for negative feedback

    Returns:
        Feedback ID or None on failure
    """
    from ..context_engine import get_feedback_collector

    try:
        collector = get_feedback_collector()

        if feedback_type == "thumbs_up":
            return collector.thumbs_up(session_id, query, response, chunk_ids, user_id)
        elif feedback_type == "thumbs_down":
            return collector.thumbs_down(session_id, query, response, chunk_ids, user_id, reason)
        elif feedback_type == "rating" and score is not None:
            return collector.rating(session_id, query, response, score, chunk_ids, user_id)
        else:
            logger.warning(f"Invalid feedback type: {feedback_type}")
            return None
    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}")
        return None


def get_analytics(days: int = 30) -> Dict[str, Any]:
    """
    Get feedback and retrieval analytics.

    Args:
        days: Number of days to include in analytics

    Returns:
        Dict with feedback stats, retrieval stats, and improvement opportunities
    """
    from ..context_engine import get_feedback_collector

    try:
        collector = get_feedback_collector()
        analytics = collector.get_analytics(days)
        analytics["improvement_opportunities"] = collector.get_improvement_opportunities(days)
        return analytics
    except Exception as e:
        logger.error(f"Failed to get analytics: {e}")
        return {"error": str(e)}
