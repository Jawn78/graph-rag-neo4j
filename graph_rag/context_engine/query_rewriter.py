"""
Query rewriting and expansion for context-aware retrieval.

Handles:
- Coreference resolution ("it", "that", "this" -> actual entities)
- Query expansion with synonyms and context
- Disambiguation using conversation history
"""

import re
import logging
from typing import List, Optional, Tuple

from .types import SessionContext, Intent

logger = logging.getLogger(__name__)

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


def rewrite_query_llm(query: str, session: Optional[SessionContext]) -> Tuple[str, List[str]]:
    """
    LLM-based query rewriting.

    Uses the chat model to:
    - Resolve coreferences using conversation context
    - Expand the query for better retrieval
    - Extract key entities
    """
    from ..config import chat_client, CHAT_MODEL

    conversation_context = ""
    if session:
        conversation_context = session.get_conversation_text(5)

    system_prompt = """You are a query rewriter for a search system. Your job is to:

1. Resolve any pronouns or references (it, this, that, they) using conversation context
2. Expand abbreviations
3. Make the query self-contained (understandable without conversation history)
4. Keep the query concise but complete

Respond in this exact format:
QUERY: <rewritten query>
ENTITIES: <comma-separated list of key entities>

If no rewriting is needed, return the original query unchanged."""

    user_prompt = f"Original query: {query}"
    if conversation_context:
        user_prompt = f"Conversation history:\n{conversation_context}\n\n{user_prompt}"

    try:
        response = chat_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=200
        )

        result = response.choices[0].message.content.strip()

        # Parse response
        rewritten = query
        entities = []

        for line in result.split("\n"):
            if line.startswith("QUERY:"):
                rewritten = line[6:].strip()
            elif line.startswith("ENTITIES:"):
                entity_str = line[9:].strip()
                if entity_str and entity_str.lower() != "none":
                    entities = [e.strip() for e in entity_str.split(",") if e.strip()]

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
    # Quick check: if no coreferences and short query, skip rewriting
    if not _has_coreference(query) and len(query.split()) < 10:
        entities = _extract_entities_simple(query)
        expanded = _expand_abbreviations(query)
        return expanded, entities

    if use_llm:
        return rewrite_query_llm(query, session)
    else:
        return rewrite_query_simple(query, session)


def build_search_query(rewritten: str, entities: List[str], intent: Intent) -> str:
    """
    Build an optimized search query based on rewritten query and intent.

    Combines the rewritten query with extracted entities for hybrid search.
    """
    parts = [rewritten]

    # Add entities as additional search terms
    for entity in entities[:3]:  # Top 3 entities
        if entity.lower() not in rewritten.lower():
            parts.append(entity)

    # Intent-specific modifications
    if intent == Intent.HOW_TO:
        # Add procedural keywords
        parts.extend(["steps", "procedure", "process"])

    elif intent == Intent.COMPARISON:
        # Add comparison keywords
        parts.extend(["difference", "comparison", "versus"])

    elif intent == Intent.DEFINITION:
        # Add definition keywords
        parts.extend(["definition", "meaning", "overview"])

    return " ".join(parts)
