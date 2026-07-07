"""
Query rewriting and expansion for context-aware retrieval.

Handles:
- Coreference resolution ("it", "that", "this" -> actual entities)
- Query expansion with synonyms and context
- Disambiguation using conversation history
"""

import re
import json
import logging
from typing import List, Optional, Tuple

from .types import SessionContext

logger = logging.getLogger(__name__)

# A query this short is unlikely to stand on its own in a conversation
FOLLOWUP_MAX_WORDS = 6

# Pronouns and references that need resolution
_COREFERENCE_PATTERNS = [
    re.compile(r"\b(it|its|it\'s)\b", re.IGNORECASE),
    re.compile(r"\b(this|that|these|those)\b", re.IGNORECASE),
    re.compile(r"\b(they|them|their|theirs)\b", re.IGNORECASE),
    re.compile(r"\b(he|him|his|she|her|hers)\b", re.IGNORECASE),
    re.compile(r"\b(the\s+same)\b", re.IGNORECASE),
    re.compile(r"\b(the\s+above|the\s+previous)\b", re.IGNORECASE),
]

# Common abbreviations to expand
_ABBREVIATIONS = {
    "pls": "please",
    "plz": "please",
    "thx": "thanks",
    "ty": "thank you",
    "bc": "because",
    "b/c": "because",
    "w/": "with",
    "w/o": "without",
    "info": "information",
    "govt": "government",
    "gov": "government",
    "dept": "department",
    "mgmt": "management",
    "mgr": "manager",
    "admin": "administration",
    "asap": "as soon as possible",
    "fyi": "for your information",
    "re": "regarding",
}


def _has_coreference(query: str) -> bool:
    """Check if query contains pronouns or references needing resolution."""
    for pattern in _COREFERENCE_PATTERNS:
        if pattern.search(query):
            return True
    return False


def _content_tokens(text: str) -> set:
    """Lowercase alphanumeric tokens of 3+ chars, for overlap heuristics."""
    return {t for t in re.findall(r"[a-z0-9]{3,}", (text or "").lower())}


def detect_followup(query: str, session: Optional[SessionContext]) -> bool:
    """
    Decide whether a query is a follow-up to the conversation rather than a
    self-contained question.

    Signals (any one suffices, given prior turns exist):
    - contains an unresolved reference (it/that/they/the above ...)
    - very short query (unlikely to stand alone)
    - shares no content words with anything -> relies on context implicitly
      is NOT used; instead we require an explicit signal to avoid false
      positives on genuinely new short questions with entities of their own.
    """
    if session is None or not session.turns:
        return False

    if _has_coreference(query):
        return True

    words = query.split()
    if len(words) <= FOLLOWUP_MAX_WORDS and not _extract_entities_simple(query):
        return True

    return False


def detect_reformulation(query: str, session: Optional[SessionContext],
                         min_overlap: float = 0.6) -> Optional[str]:
    """
    Detect whether this query is a reformulation of the previous user query
    (an implicit signal that the previous answer missed the mark).

    Returns the previous query if it is a reformulation, else None.
    """
    if session is None:
        return None

    prev = session.get_last_user_query()
    if not prev or prev.strip().lower() == query.strip().lower():
        return None

    cur_toks = _content_tokens(query)
    prev_toks = _content_tokens(prev)
    if not cur_toks or not prev_toks:
        return None

    # Overlap coefficient (not Jaccard): reformulations often change word
    # forms and add qualifiers, which Jaccard punishes too hard.
    overlap = len(cur_toks & prev_toks) / min(len(cur_toks), len(prev_toks))
    return prev if overlap >= min_overlap else None


def _expand_abbreviations(query: str) -> str:
    """Expand common abbreviations in query."""
    words = query.split()
    expanded = []
    for word in words:
        # Preserve case for expansion
        lower = word.lower().rstrip(".,!?")
        if lower in _ABBREVIATIONS:
            expanded.append(_ABBREVIATIONS[lower])
        else:
            expanded.append(word)
    return " ".join(expanded)


def _extract_entities_simple(text: str) -> List[str]:
    """
    Simple entity extraction using capitalization and noun phrases.

    For production, consider spaCy or a dedicated NER model.
    """
    entities = []

    # Extract capitalized sequences (likely proper nouns)
    cap_pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')
    entities.extend(cap_pattern.findall(text))

    # Extract quoted strings
    quote_pattern = re.compile(r'"([^"]+)"|\'([^\']+)\'')
    for match in quote_pattern.finditer(text):
        entities.append(match.group(1) or match.group(2))

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for e in entities:
        if e.lower() not in seen and len(e) > 2:
            unique.append(e)
            seen.add(e.lower())

    return unique[:10]  # Limit to top 10


def rewrite_query_simple(query: str, session: Optional[SessionContext]) -> Tuple[str, List[str]]:
    """
    Simple query rewriting without LLM.

    Uses rule-based coreference resolution and abbreviation expansion.
    Returns (rewritten_query, extracted_entities).
    """
    # Expand abbreviations
    query = _expand_abbreviations(query)

    entities = []

    # Check if coreference resolution is needed
    if session and _has_coreference(query):
        # Get entities from recent conversation
        recent_turns = session.get_recent_turns(3)
        for turn in recent_turns:
            if turn.entities:
                entities.extend(turn.entities)

        # Get entities from session entity map
        entities.extend(list(session.entities.keys())[:5])

        # Get current topic
        topic = session.get_current_topic()
        if topic:
            entities.insert(0, topic)

        # Simple replacement: replace "it" with most recent entity
        if entities:
            primary_entity = entities[0]

            # Replace pronouns with entity (simple approach)
            query = re.sub(r'\b(it|this|that)\b', primary_entity, query, flags=re.IGNORECASE, count=1)

            logger.debug(f"Resolved coreference: '{primary_entity}'")

    # Extract entities from current query
    query_entities = _extract_entities_simple(query)
    entities.extend(query_entities)

    # Deduplicate entities
    seen = set()
    unique_entities = []
    for e in entities:
        if e.lower() not in seen:
            unique_entities.append(e)
            seen.add(e.lower())

    return query, unique_entities[:10]


def _parse_rewrite_response(raw: str, original_query: str) -> Tuple[str, List[str]]:
    """
    Parse the LLM rewrite response (expected JSON) defensively.

    A mangled rewrite is worse than none, so any parse/validation failure
    returns the original query untouched.
    """
    text = (raw or "").strip()

    # Strip markdown code fences if the model added them
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()

    # Grab the first JSON object in the output
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return original_query, []

    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return original_query, []

    rewritten = obj.get("query")
    entities = obj.get("entities") or []

    if not isinstance(rewritten, str) or not rewritten.strip():
        rewritten = original_query
    # Reject runaway rewrites (model rambling instead of rewriting)
    elif len(rewritten) > max(200, 4 * len(original_query)):
        rewritten = original_query

    if not isinstance(entities, list):
        entities = []
    entities = [str(e).strip() for e in entities if str(e).strip()][:10]

    return rewritten.strip(), entities


_REWRITE_SYSTEM_PROMPT = """You rewrite search queries to be self-contained.

Rules:
1. Resolve pronouns/references (it, this, that, they) using the conversation.
2. Expand abbreviations.
3. Keep the rewrite concise; do not answer the question.
4. If no rewriting is needed, return the query unchanged.

Respond with ONLY a JSON object: {"query": "<rewritten query>", "entities": ["<entity>", ...]}

Examples:
Conversation:
User: What does the dental plan cover?
Query: does it cover implants
{"query": "does the dental plan cover implants", "entities": ["dental plan", "implants"]}

Conversation:
User: Summarize Form DD-214.
Query: how do I request a copy
{"query": "how do I request a copy of Form DD-214", "entities": ["Form DD-214"]}"""


def rewrite_query_llm(query: str, session: Optional[SessionContext]) -> Tuple[str, List[str]]:
    """
    LLM-based query rewriting with JSON output and safe fallback.

    Resolves coreferences using conversation context and extracts entities.
    Falls back to rule-based rewriting on any failure.
    """
    from ..config import chat_client, CHAT_MODEL

    conversation_context = ""
    if session:
        conversation_context = session.get_conversation_text(5)
        summary = session.metadata.get("summary")
        if summary:
            conversation_context = f"(Earlier: {summary})\n{conversation_context}"

    user_prompt = f"Query: {query}"
    if conversation_context:
        user_prompt = f"Conversation:\n{conversation_context}\n\n{user_prompt}"

    try:
        response = chat_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=200
        )

        raw = response.choices[0].message.content or ""
        rewritten, entities = _parse_rewrite_response(raw, query)

        if rewritten == query and not entities:
            # Model produced nothing usable; rules may still resolve something
            return rewrite_query_simple(query, session)

        logger.debug(f"LLM rewrite: '{query[:30]}...' -> '{rewritten[:30]}...'")
        return rewritten, entities

    except Exception as e:
        logger.warning(f"LLM query rewriting failed: {e}")
        # Fall back to simple rewriting
        return rewrite_query_simple(query, session)


def rewrite_query(query: str, session: Optional[SessionContext] = None,
                  use_llm: bool = False) -> Tuple[str, List[str]]:
    """
    Rewrite a query for better retrieval.

    Args:
        query: The original user query
        session: Optional session context for conversation history
        use_llm: Whether to use LLM for rewriting (slower but better)

    Returns:
        (rewritten_query, extracted_entities) tuple
    """
    # Fast path: self-contained queries need no LLM and no coreference work.
    # The LLM (when enabled) is reserved for genuine follow-ups, where it
    # actually earns its latency.
    if not _has_coreference(query) and not detect_followup(query, session):
        entities = _extract_entities_simple(query)
        expanded = _expand_abbreviations(query)
        return expanded, entities

    if use_llm:
        return rewrite_query_llm(query, session)
    else:
        return rewrite_query_simple(query, session)
