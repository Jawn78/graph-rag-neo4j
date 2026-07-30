"""
Intent classification for query understanding.

Uses a combination of rule-based patterns and LLM-based classification
to determine the user's intent from their query.
"""

import os
import re
import logging
from typing import Optional, Tuple

from .types import Intent, SessionContext

logger = logging.getLogger(__name__)

# Whether to use LLM for intent classification (slower but more accurate)
USE_LLM_INTENT = os.getenv("USE_LLM_INTENT", "0") == "1"


# Rule-based patterns for fast intent detection
_INTENT_PATTERNS = {
    Intent.HOW_TO: [
        re.compile(r"^how\s+(do|can|should|would|to)\b", re.IGNORECASE),
        re.compile(r"^what\s+(are\s+)?the\s+steps\b", re.IGNORECASE),
        re.compile(r"\bstep[\s-]by[\s-]step\b", re.IGNORECASE),
        re.compile(r"^(explain|show|walk)\s+(me\s+)?how\b", re.IGNORECASE),
    ],
    Intent.COMPARISON: [
        re.compile(r"\bdifference\s+between\b", re.IGNORECASE),
        re.compile(r"\bcompare\b", re.IGNORECASE),
        re.compile(r"\bvs\.?\b", re.IGNORECASE),
        re.compile(r"\bversus\b", re.IGNORECASE),
        re.compile(r"\bbetter\s+(than|or)\b", re.IGNORECASE),
        re.compile(r"\bwhich\s+(is|one)\s+(better|best)\b", re.IGNORECASE),
    ],
    Intent.SUMMARIZATION: [
        re.compile(r"^summarize\b", re.IGNORECASE),
        re.compile(r"^(give|provide)\s+(me\s+)?a\s+summary\b", re.IGNORECASE),
        re.compile(r"\bin\s+brief\b", re.IGNORECASE),
        re.compile(r"\btl;?dr\b", re.IGNORECASE),
        re.compile(r"^overview\s+of\b", re.IGNORECASE),
    ],
    Intent.DEFINITION: [
        re.compile(r"^(what\s+is|what\'s|whats)\s+(a|an|the)?\s*\w+\??$", re.IGNORECASE),
        re.compile(r"^define\b", re.IGNORECASE),
        re.compile(r"\bdefinition\s+of\b", re.IGNORECASE),
        re.compile(r"^meaning\s+of\b", re.IGNORECASE),
    ],
    Intent.LIST: [
        re.compile(r"^list\b", re.IGNORECASE),
        re.compile(r"^(what|which)\s+are\s+(the|all)\b", re.IGNORECASE),
        re.compile(r"\benumerate\b", re.IGNORECASE),
        re.compile(r"^(give|show|provide)\s+(me\s+)?(a\s+)?list\b", re.IGNORECASE),
    ],
    Intent.CLARIFICATION: [
        re.compile(r"^what\s+do\s+you\s+mean\b", re.IGNORECASE),
        re.compile(r"^can\s+you\s+(explain|clarify)\b", re.IGNORECASE),
        re.compile(r"^(I\s+don\'t\s+understand|unclear)\b", re.IGNORECASE),
        re.compile(r"^(what|which)\s+(is|are)\s+(that|it|this)\b", re.IGNORECASE),
    ],
    Intent.EXPLORATORY: [
        re.compile(r"^tell\s+me\s+(about|more)\b", re.IGNORECASE),
        re.compile(r"^(I\'d\s+like\s+to\s+know|I\s+want\s+to\s+learn)\b", re.IGNORECASE),
        re.compile(r"^(explain|describe)\b", re.IGNORECASE),
    ],
}

# Keywords that suggest factual intent (fallback)
_FACTUAL_KEYWORDS = [
    r"^what\b", r"^who\b", r"^when\b", r"^where\b", r"^which\b",
    r"^does\b", r"^is\b", r"^are\b", r"^can\b", r"^will\b",
]
_FACTUAL_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _FACTUAL_KEYWORDS]


# When several intents' patterns fire (e.g. "What is X?" is both DEFINITION
# and FACTUAL-shaped), prefer the more specific intent. Order matters.
_INTENT_SPECIFICITY = [
    Intent.COMPARISON,
    Intent.HOW_TO,
    Intent.SUMMARIZATION,
    Intent.LIST,
    Intent.CLARIFICATION,
    Intent.DEFINITION,
    Intent.EXPLORATORY,
]


def classify_intent_rules(query: str) -> Tuple[Intent, float]:
    """
    Classify intent using rule-based patterns.

    Scores ALL intents rather than returning on the first pattern hit, so
    overlapping patterns don't get resolved by dict iteration order. Ties are
    broken by specificity (comparison beats factual, etc.).

    Returns (intent, confidence) tuple.
    """
    query = query.strip()
    if not query:
        return Intent.UNKNOWN, 0.0

    # Count pattern hits per intent
    hits = {
        intent: sum(1 for pattern in patterns if pattern.search(query))
        for intent, patterns in _INTENT_PATTERNS.items()
    }
    matched = {intent: n for intent, n in hits.items() if n > 0}

    if matched:
        best = max(
            matched,
            key=lambda i: (matched[i], -_INTENT_SPECIFICITY.index(i)),
        )
        # More corroborating patterns -> higher confidence, capped at 0.95
        confidence = min(0.95, 0.8 + 0.05 * matched[best])
        return best, confidence

    # Check for factual patterns (most common fallback)
    for pattern in _FACTUAL_PATTERNS:
        if pattern.search(query):
            return Intent.FACTUAL, 0.7

    # Default to exploratory for statements, factual for questions
    if query.endswith("?"):
        return Intent.FACTUAL, 0.5
    else:
        return Intent.EXPLORATORY, 0.4


def classify_intent_llm(query: str, conversation_context: str = "") -> Tuple[Intent, float]:
    """
    Classify intent using LLM few-shot prompting.

    More accurate but slower than rule-based classification.
    """
    from ..config import chat_client, CHAT_MODEL

    system_prompt = """You are an intent classifier. Given a user query, classify it into exactly one of these categories:

- factual: Direct fact lookup ("What is X?", "When did Y happen?")
- comparison: Compare entities ("What's the difference between X and Y?")
- how_to: Procedural/steps ("How do I X?", "Steps to Y")
- summarization: Summarize content ("Summarize X", "TL;DR of Y")
- clarification: Follow-up on previous answer ("What do you mean?", "Can you explain that?")
- exploratory: Open-ended exploration ("Tell me about X")
- list: Enumeration request ("List all X", "What are the types of Y?")
- definition: Define a term ("What is X?", "Define Y")

Respond with ONLY the category name, nothing else."""

    user_prompt = f"Query: {query}"
    if conversation_context:
        user_prompt = f"Recent conversation:\n{conversation_context}\n\n{user_prompt}"

    try:
        response = chat_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=20
        )

        result = response.choices[0].message.content.strip().lower()

        # Map response to Intent
        intent_map = {
            "factual": Intent.FACTUAL,
            "comparison": Intent.COMPARISON,
            "how_to": Intent.HOW_TO,
            "summarization": Intent.SUMMARIZATION,
            "clarification": Intent.CLARIFICATION,
            "exploratory": Intent.EXPLORATORY,
            "list": Intent.LIST,
            "definition": Intent.DEFINITION,
        }

        intent = intent_map.get(result, Intent.UNKNOWN)
        confidence = 0.9 if intent != Intent.UNKNOWN else 0.3

        logger.debug(f"LLM classified '{query[:50]}...' as {intent.value} ({confidence})")
        return intent, confidence

    except Exception as e:
        logger.warning(f"LLM intent classification failed: {e}")
        # Fall back to rules
        return classify_intent_rules(query)


def classify_intent(query: str, session: Optional[SessionContext] = None,
                    use_llm: Optional[bool] = None) -> Tuple[Intent, float]:
    """
    Classify the intent of a user query.

    Args:
        query: The user's query
        session: Optional session context for conversation history
        use_llm: Override USE_LLM_INTENT setting

    Returns:
        (intent, confidence) tuple
    """
    should_use_llm = use_llm if use_llm is not None else USE_LLM_INTENT

    if should_use_llm:
        conversation = session.get_conversation_text(3) if session else ""
        return classify_intent_llm(query, conversation)
    else:
        return classify_intent_rules(query)


def get_intent_hints(intent: Intent) -> dict:
    """
    Get retrieval hints based on intent.

    Returns a dict of parameters to adjust the retrieval strategy.
    """
    hints = {
        "require_recency": False,
        "require_comparison": False,
        "require_procedure": False,
        "diversity_weight": 0.3,
        "top_k_multiplier": 1.0,
    }

    if intent == Intent.COMPARISON:
        hints["require_comparison"] = True
        hints["diversity_weight"] = 0.6  # Need diverse sources
        hints["top_k_multiplier"] = 1.5  # Get more candidates

    elif intent == Intent.HOW_TO:
        hints["require_procedure"] = True
        hints["diversity_weight"] = 0.1  # Prefer coherent steps

    elif intent == Intent.SUMMARIZATION:
        hints["top_k_multiplier"] = 2.0  # Need more context

    elif intent == Intent.LIST:
        hints["diversity_weight"] = 0.5  # Need coverage
        hints["top_k_multiplier"] = 1.5

    elif intent == Intent.CLARIFICATION:
        hints["require_recency"] = True  # Focus on recent context

    return hints
