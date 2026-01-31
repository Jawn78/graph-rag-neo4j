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
