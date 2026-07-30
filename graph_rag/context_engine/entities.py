"""
Entity extraction and resolution for context-aware retrieval.

Implements:
- Pattern-based named entity recognition
- Entity linking to knowledge base
- Coreference resolution helpers
- Entity-based query expansion
"""

import re
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of named entities."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    MONEY = "money"
    PERCENT = "percent"
    PRODUCT = "product"
    EVENT = "event"
    POLICY = "policy"
    DOCUMENT = "document"
    UNKNOWN = "unknown"


@dataclass
class Entity:
    """A recognized named entity."""
    text: str
    type: EntityType
    start: int
    end: int
    confidence: float = 1.0
    canonical: Optional[str] = None  # Normalized/linked form
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash((self.text.lower(), self.type))

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self.text.lower() == other.text.lower() and self.type == other.type


# Pattern-based entity extractors
_ENTITY_PATTERNS = {
    EntityType.DATE: [
        # Full dates
        re.compile(r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b"),
        re.compile(r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b"),
        # Month names
        re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b", re.IGNORECASE),
        re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4}\b", re.IGNORECASE),
        # Relative dates
        re.compile(r"\b(last|next|this)\s+(week|month|year|quarter)\b", re.IGNORECASE),
        re.compile(r"\b(Q[1-4])\s+\d{4}\b", re.IGNORECASE),
        re.compile(r"\bFY\s*\d{2,4}\b", re.IGNORECASE),
    ],
    EntityType.MONEY: [
        re.compile(r"\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|M|B|K))?\b"),
        re.compile(r"\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars|USD|EUR|GBP)\b", re.IGNORECASE),
    ],
    EntityType.PERCENT: [
        re.compile(r"\b\d+(?:\.\d+)?%\b"),
        re.compile(r"\b\d+(?:\.\d+)?\s*percent\b", re.IGNORECASE),
    ],
    EntityType.ORGANIZATION: [
        # Common org suffixes
        re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|LLC|Ltd|Company|Co|Foundation|Institute|Association|Department|Agency|Bureau|Office|Commission)\.?)\b"),
        # Acronyms (3+ caps)
        re.compile(r"\b([A-Z]{3,})\b"),
    ],
    EntityType.POLICY: [
        # Policy/regulation patterns
        re.compile(r"\b(Policy\s+\d+[-.]?\d*)\b", re.IGNORECASE),
        re.compile(r"\b(Section\s+\d+(?:\.\d+)*)\b", re.IGNORECASE),
        re.compile(r"\b(Article\s+[IVXLCDM]+|\d+)\b", re.IGNORECASE),
        re.compile(r"\b(Regulation\s+[A-Z]?[-]?\d+)\b", re.IGNORECASE),
    ],
    EntityType.DOCUMENT: [
        # Document references
        re.compile(r"\b(Form\s+\d+[-A-Z]*)\b", re.IGNORECASE),
        re.compile(r"\b(Appendix\s+[A-Z])\b", re.IGNORECASE),
        re.compile(r"\b(Exhibit\s+\d+)\b", re.IGNORECASE),
        re.compile(r"\b(Schedule\s+[A-Z])\b", re.IGNORECASE),
    ],
}

# Common titles that precede person names
_PERSON_TITLES = {"Mr", "Mrs", "Ms", "Dr", "Prof", "Sir", "Madam", "Director", "Manager", "CEO", "CFO", "CTO"}

# Stopwords that shouldn't be entities
_ENTITY_STOPWORDS = {
    "the", "a", "an", "this", "that", "these", "those", "it", "they",
    "what", "which", "who", "where", "when", "how", "why",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must",
    "and", "or", "but", "if", "then", "because", "while",
}


def extract_entities(text: str, types: Optional[List[EntityType]] = None) -> List[Entity]:
    """
    Extract named entities from text using pattern matching.

    Args:
        text: The text to extract entities from
        types: Optional list of entity types to extract (all if None)

    Returns:
        List of extracted entities
    """
    if not text:
        return []

    entities: List[Entity] = []
    seen_spans: Set[Tuple[int, int]] = set()

    # Apply pattern extractors
    for entity_type, patterns in _ENTITY_PATTERNS.items():
        if types and entity_type not in types:
            continue

        for pattern in patterns:
            for match in pattern.finditer(text):
                start, end = match.span()

                # Skip overlapping spans
                if any(s <= start < e or s < end <= e for s, e in seen_spans):
                    continue

                entity_text = match.group(0).strip()

                # Skip stopwords
                if entity_text.lower() in _ENTITY_STOPWORDS:
                    continue

                entities.append(Entity(
                    text=entity_text,
                    type=entity_type,
                    start=start,
                    end=end,
                    confidence=0.8
                ))
                seen_spans.add((start, end))

    # Extract capitalized sequences (potential names/orgs)
    if not types or EntityType.PERSON in types or EntityType.ORGANIZATION in types:
        cap_pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
        for match in cap_pattern.finditer(text):
            start, end = match.span()

            # Skip if overlapping
            if any(s <= start < e or s < end <= e for s, e in seen_spans):
                continue

            entity_text = match.group(0)

            # Skip short or stopword matches
            if len(entity_text) < 4 or entity_text.lower() in _ENTITY_STOPWORDS:
                continue

            # Determine if person or organization
            words = entity_text.split()
            if words[0] in _PERSON_TITLES:
                entity_type = EntityType.PERSON
            elif len(words) >= 2 and len(words) <= 4:
                # Likely a person name
                entity_type = EntityType.PERSON
            else:
                entity_type = EntityType.ORGANIZATION

            entities.append(Entity(
                text=entity_text,
                type=entity_type,
                start=start,
                end=end,
                confidence=0.6
            ))
            seen_spans.add((start, end))

    # Sort by position
    entities.sort(key=lambda e: e.start)

    return entities


def extract_entity_strings(text: str) -> List[str]:
    """
    Extract entity strings from text.

    Simple wrapper that returns just the text of extracted entities.
    """
    entities = extract_entities(text)
    return [e.text for e in entities]


def resolve_coreferences(text: str, entities: List[Entity],
                         prior_entities: List[str]) -> Dict[str, str]:
    """
    Resolve coreferences (pronouns) to entities.

    Args:
        text: The text containing potential coreferences
        entities: Entities extracted from current text
        prior_entities: Entities from prior context

    Returns:
        Dict mapping pronouns/references to resolved entities
    """
    resolutions: Dict[str, str] = {}

    # Combine entities
    all_entities = [e.text for e in entities] + prior_entities

    if not all_entities:
        return resolutions

    # Find pronouns and references
    pronoun_pattern = re.compile(r'\b(it|this|that|they|them|these|those|the above|the same)\b', re.IGNORECASE)

    for match in pronoun_pattern.finditer(text):
        pronoun = match.group(0).lower()

        # Simple heuristic: resolve to most recent entity
        if pronoun in ("it", "this", "that"):
            # Singular - use most recent singular entity
            for entity in reversed(all_entities):
                if not entity.lower().endswith("s"):  # Simple plural check
                    resolutions[pronoun] = entity
                    break
            if pronoun not in resolutions and all_entities:
                resolutions[pronoun] = all_entities[0]

        elif pronoun in ("they", "them", "these", "those"):
            # Plural - use most recent plural or organization
            for entity in reversed(all_entities):
                if entity.lower().endswith("s") or entity.isupper():
                    resolutions[pronoun] = entity
                    break
            if pronoun not in resolutions and all_entities:
                resolutions[pronoun] = all_entities[0]

        elif pronoun in ("the above", "the same"):
            if all_entities:
                resolutions[pronoun] = all_entities[0]

    return resolutions


def link_entities(entities: List[Entity],
                  knowledge_base: Optional[Dict[str, str]] = None) -> List[Entity]:
    """
    Link entities to canonical forms in knowledge base.

    Args:
        entities: Extracted entities
        knowledge_base: Optional dict mapping variants to canonical forms

    Returns:
        Entities with canonical field populated
    """
    if not knowledge_base:
        # Use entity text as canonical
        for entity in entities:
            entity.canonical = entity.text
        return entities

    for entity in entities:
        text_lower = entity.text.lower()

        # Try exact match
        if text_lower in knowledge_base:
            entity.canonical = knowledge_base[text_lower]
            continue

        # Try partial match
        for variant, canonical in knowledge_base.items():
            if variant in text_lower or text_lower in variant:
                entity.canonical = canonical
                break
        else:
            entity.canonical = entity.text

    return entities


def expand_query_with_entities(query: str, entities: List[Entity]) -> str:
    """
    Expand query by adding entity-related terms.

    Useful for improving retrieval when entities are recognized.
    """
    if not entities:
        return query

    expansion_terms = []

    for entity in entities:
        # Add canonical form if different
        if entity.canonical and entity.canonical != entity.text:
            expansion_terms.append(entity.canonical)

        # Add type-specific expansions
        if entity.type == EntityType.POLICY:
            expansion_terms.extend(["policy", "regulation", "requirement"])
        elif entity.type == EntityType.ORGANIZATION:
            expansion_terms.append("organization")
        elif entity.type == EntityType.DATE:
            expansion_terms.extend(["date", "when"])

    if expansion_terms:
        return f"{query} {' '.join(set(expansion_terms))}"

    return query


def group_entities_by_type(entities: List[Entity]) -> Dict[EntityType, List[Entity]]:
    """Group entities by their type."""
    groups: Dict[EntityType, List[Entity]] = {}
    for entity in entities:
        if entity.type not in groups:
            groups[entity.type] = []
        groups[entity.type].append(entity)
    return groups
