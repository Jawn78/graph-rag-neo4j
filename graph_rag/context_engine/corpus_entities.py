"""
Corpus-derived entity dictionary.

Generic NER keys off capitalization, but real users type lowercase
("what does hipaa cover"), and the entities that matter for retrieval are the
ones that actually appear in the ingested corpus. This module builds a
case-insensitive dictionary of document titles and section headings from
Neo4j and matches queries against it.

Matched entities are used to enrich QueryContext.entities and to strengthen
the keyword (FTS) leg of hybrid search. They are deliberately NOT injected
into the text that gets embedded — appending terms distorts dense retrieval.
"""

import os
import re
import time
import logging
import threading
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Reload the dictionary from Neo4j after this many seconds
CORPUS_ENTITY_TTL_SECONDS = int(os.getenv("CORPUS_ENTITY_TTL_SECONDS", "600"))

# Cap on entries loaded per source (titles / headings)
CORPUS_ENTITY_LIMIT = int(os.getenv("CORPUS_ENTITY_LIMIT", "5000"))

# Ignore very short aliases: they match everything
_MIN_ALIAS_LEN = 4

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-/&\.]*")


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace for alias matching."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


class CorpusEntityDictionary:
    """
    Case-insensitive alias -> canonical entity map built from the corpus.

    Aliases come from Document titles and Chunk headings. Matching is
    longest-alias-first over the normalized query text.
    """

    def __init__(self, ttl_seconds: int = CORPUS_ENTITY_TTL_SECONDS):
        self._aliases: Dict[str, str] = {}   # normalized alias -> canonical form
        self._loaded_at: float = 0.0
        self._ttl = ttl_seconds
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_entries(self, entries: List[str]) -> None:
        """Load alias entries directly (also used by tests)."""
        aliases: Dict[str, str] = {}
        for raw in entries:
            canonical = re.sub(r"\s+", " ", (raw or "").strip())
            norm = _normalize(canonical)
            if len(norm) < _MIN_ALIAS_LEN:
                continue
            # First occurrence wins (titles are loaded before headings)
            aliases.setdefault(norm, canonical)

            # Also index a filename-ish variant without separators:
            # "benefits_overview_2024" -> "benefits overview 2024"
            despaced = norm.replace("_", " ").replace("-", " ")
            despaced = re.sub(r"\s+", " ", despaced).strip()
            if despaced != norm and len(despaced) >= _MIN_ALIAS_LEN:
                aliases.setdefault(despaced, canonical)

        with self._lock:
            self._aliases = aliases
            self._loaded_at = time.time()

        logger.info(f"Corpus entity dictionary loaded: {len(aliases)} aliases")

    def refresh_from_neo4j(self, driver) -> None:
        """(Re)load titles and headings from the graph."""
        from neo4j import READ_ACCESS
        from ..config import NEO4J_DB

        entries: List[str] = []
        try:
            with driver.session(database=NEO4J_DB, default_access_mode=READ_ACCESS) as s:
                titles = s.run(
                    "MATCH (d:Document) WHERE d.title IS NOT NULL AND d.title <> '' "
                    "RETURN DISTINCT d.title AS t LIMIT $lim",
                    lim=CORPUS_ENTITY_LIMIT,
                ).data()
                entries.extend(r["t"] for r in titles)

                headings = s.run(
                    "MATCH (c:Chunk) WHERE c.heading IS NOT NULL AND c.heading <> '' "
                    "RETURN DISTINCT c.heading AS h LIMIT $lim",
                    lim=CORPUS_ENTITY_LIMIT,
                ).data()
                entries.extend(r["h"] for r in headings)
        except Exception as e:
            logger.warning(f"Corpus entity load failed (keeping previous dictionary): {e}")
            return

        self.load_entries(entries)

    def ensure_fresh(self, driver) -> None:
        """Load or reload the dictionary if it's stale."""
        if driver is None:
            return
        if time.time() - self._loaded_at < self._ttl and self._aliases:
            return
        self.refresh_from_neo4j(driver)

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match(self, query: str, max_matches: int = 5) -> List[str]:
        """
        Return canonical entities whose alias appears in the query
        (case-insensitive), longest alias first so specific entries beat
        their substrings.
        """
        norm_query = _normalize(query)
        if not norm_query or not self._aliases:
            return []

        padded = f" {norm_query} "
        matches: List[str] = []
        for alias in sorted(self._aliases, key=len, reverse=True):
            if len(matches) >= max_matches:
                break
            # Word-boundary containment: alias must not be mid-word
            if f" {alias} " in padded or padded.strip() == alias:
                canonical = self._aliases[alias]
                if canonical not in matches:
                    matches.append(canonical)
                    # Consume so substrings of this alias don't also match
                    padded = padded.replace(f" {alias} ", " ")
        return matches


# Global instance (dictionary is corpus-wide, not per-session)
_dictionary: Optional[CorpusEntityDictionary] = None
_dict_lock = threading.Lock()


def get_corpus_dictionary() -> CorpusEntityDictionary:
    """Get the global corpus entity dictionary."""
    global _dictionary
    if _dictionary is None:
        with _dict_lock:
            if _dictionary is None:
                _dictionary = CorpusEntityDictionary()
    return _dictionary


def match_corpus_entities(driver, query: str, max_matches: int = 5) -> List[str]:
    """
    Match a query against the corpus entity dictionary, refreshing it from
    Neo4j when stale. Best-effort: returns [] on any failure.
    """
    try:
        d = get_corpus_dictionary()
        d.ensure_fresh(driver)
        return d.match(query, max_matches=max_matches)
    except Exception as e:
        logger.debug(f"Corpus entity matching failed: {e}")
        return []
