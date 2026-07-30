"""
Metadata filtering for context-aware retrieval.

Implements:
- Temporal filtering (recency weighting)
- Source type filtering
- Document attribute filtering
- Combined filter strategies
"""

import os
import re
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Temporal filtering configuration
RECENCY_DECAY_DAYS = int(os.getenv("RECENCY_DECAY_DAYS", "365"))
RECENCY_WEIGHT_MIN = float(os.getenv("RECENCY_WEIGHT_MIN", "0.5"))


@dataclass
class FilterCriteria:
    """
    Criteria for filtering and weighting chunks.

    All criteria are optional and combined with AND logic.
    """
    # Temporal filters
    require_recency: bool = False
    max_age_days: Optional[int] = None
    min_date: Optional[datetime] = None
    max_date: Optional[datetime] = None

    # Source filters
    allowed_sources: List[str] = field(default_factory=list)
    blocked_sources: List[str] = field(default_factory=list)
    source_pattern: Optional[str] = None

    # Document type filters
    allowed_extensions: List[str] = field(default_factory=list)
    blocked_extensions: List[str] = field(default_factory=list)

    # Content filters
    required_keywords: List[str] = field(default_factory=list)
    blocked_keywords: List[str] = field(default_factory=list)

    # Metadata key-value filters
    metadata_filters: Dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        """Check if any filters are set."""
        return (
            not self.require_recency and
            self.max_age_days is None and
            self.min_date is None and
            self.max_date is None and
            not self.allowed_sources and
            not self.blocked_sources and
            self.source_pattern is None and
            not self.allowed_extensions and
            not self.blocked_extensions and
            not self.required_keywords and
            not self.blocked_keywords and
            not self.metadata_filters
        )


def _parse_date_from_metadata(metadata: Dict[str, Any]) -> Optional[datetime]:
    """
    Extract date from chunk/document metadata.

    Looks for common date fields and parses various formats.
    """
    date_fields = ["date", "created", "modified", "published", "timestamp", "created_at"]

    for field_name in date_fields:
        value = metadata.get(field_name)
        if not value:
            continue

        if isinstance(value, datetime):
            return value

        if isinstance(value, str):
            # Try common date formats
            formats = [
                "%Y-%m-%d",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y/%m/%d",
                "%d/%m/%Y",
                "%m/%d/%Y",
                "%B %d, %Y",
                "%b %d, %Y",
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(value[:len(fmt) + 5], fmt)
                except (ValueError, IndexError):
                    continue

    return None


def _extract_date_from_path(path: str) -> Optional[datetime]:
    """
    Extract date from file path if it contains date patterns.

    Common patterns: 2024-01-15, 2024_01_15, 20240115
    """
    patterns = [
        (r"(\d{4})-(\d{2})-(\d{2})", "%Y-%m-%d"),
        (r"(\d{4})_(\d{2})_(\d{2})", "%Y_%m_%d"),
        (r"(\d{4})(\d{2})(\d{2})", "%Y%m%d"),
    ]

    for pattern, fmt in patterns:
        match = re.search(pattern, path)
        if match:
            try:
                date_str = "-".join(match.groups())
                return datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                continue

    return None


def _get_chunk_date(chunk: Dict[str, Any]) -> Optional[datetime]:
    """Get the date associated with a chunk."""
    # Try metadata
    metadata = chunk.get("metadata", {})
    if isinstance(metadata, str):
        try:
            import json
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}

    date = _parse_date_from_metadata(metadata)
    if date:
        return date

    # Try path in metadata
    path = metadata.get("path", "") or chunk.get("path", "")
    if path:
        date = _extract_date_from_path(path)
        if date:
            return date

    return None


def _get_source_info(chunk: Dict[str, Any]) -> Dict[str, str]:
    """Extract source information from chunk."""
    metadata = chunk.get("metadata", {})
    if isinstance(metadata, str):
        try:
            import json
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}

    return {
        "source": metadata.get("source", "") or chunk.get("source", ""),
        "path": metadata.get("path", "") or chunk.get("path", ""),
        "filename": metadata.get("filename", "") or chunk.get("filename", ""),
        "ext": metadata.get("ext", "") or "",
    }


def compute_recency_weight(chunk: Dict[str, Any],
                           decay_days: int = RECENCY_DECAY_DAYS,
                           min_weight: float = RECENCY_WEIGHT_MIN) -> float:
    """
    Compute a recency-based weight for a chunk.

    Uses exponential decay based on document age.

    Args:
        chunk: The chunk to weight
        decay_days: Days for weight to decay to ~37% (1/e)
        min_weight: Minimum weight (floor)

    Returns:
        Weight between min_weight and 1.0
    """
    chunk_date = _get_chunk_date(chunk)
    if not chunk_date:
        return 1.0  # No date = no penalty

    age_days = (datetime.now() - chunk_date).days
    if age_days <= 0:
        return 1.0

    # Exponential decay: weight = e^(-age/decay)
    import math
    weight = math.exp(-age_days / decay_days)

    return max(min_weight, weight)


def apply_filters(chunks: List[Dict[str, Any]],
                  criteria: FilterCriteria) -> List[Dict[str, Any]]:
    """
    Apply filter criteria to chunks.

    Returns chunks that pass all filters.
    """
    if criteria.is_empty():
        return chunks

    result = []

    for chunk in chunks:
        if _passes_filters(chunk, criteria):
            result.append(chunk)

    logger.debug(f"Filtering: {len(chunks)} -> {len(result)} chunks")
    return result


def _passes_filters(chunk: Dict[str, Any], criteria: FilterCriteria) -> bool:
    """Check if a chunk passes all filter criteria."""
    source_info = _get_source_info(chunk)
    chunk_date = _get_chunk_date(chunk)

    # Temporal filters
    if criteria.max_age_days is not None and chunk_date:
        max_date = datetime.now() - timedelta(days=criteria.max_age_days)
        if chunk_date < max_date:
            return False

    if criteria.min_date is not None and chunk_date:
        if chunk_date < criteria.min_date:
            return False

    if criteria.max_date is not None and chunk_date:
        if chunk_date > criteria.max_date:
            return False

    # Source filters
    source = source_info["source"].lower()
    path = source_info["path"].lower()

    if criteria.allowed_sources:
        allowed_lower = [s.lower() for s in criteria.allowed_sources]
        if not any(a in source or a in path for a in allowed_lower):
            return False

    if criteria.blocked_sources:
        blocked_lower = [s.lower() for s in criteria.blocked_sources]
        if any(b in source or b in path for b in blocked_lower):
            return False

    if criteria.source_pattern:
        pattern = re.compile(criteria.source_pattern, re.IGNORECASE)
        if not pattern.search(path) and not pattern.search(source):
            return False

    # Extension filters
    ext = source_info["ext"].lower().lstrip(".")

    if criteria.allowed_extensions:
        allowed_ext = [e.lower().lstrip(".") for e in criteria.allowed_extensions]
        if ext not in allowed_ext:
            return False

    if criteria.blocked_extensions:
        blocked_ext = [e.lower().lstrip(".") for e in criteria.blocked_extensions]
        if ext in blocked_ext:
            return False

    # Content filters
    text = (chunk.get("text", "") or "").lower()
    heading = (chunk.get("heading", "") or "").lower()
    content = text + " " + heading

    if criteria.required_keywords:
        for kw in criteria.required_keywords:
            if kw.lower() not in content:
                return False

    if criteria.blocked_keywords:
        for kw in criteria.blocked_keywords:
            if kw.lower() in content:
                return False

    # Metadata filters
    metadata = chunk.get("metadata", {})
    if isinstance(metadata, str):
        try:
            import json
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}

    for key, expected in criteria.metadata_filters.items():
        actual = metadata.get(key)
        if actual != expected:
            return False

    return True


def apply_recency_boost(chunks: List[Dict[str, Any]],
                        decay_days: int = RECENCY_DECAY_DAYS,
                        min_weight: float = RECENCY_WEIGHT_MIN) -> List[Dict[str, Any]]:
    """
    Apply recency-based boosting to chunks.

    Modifies chunk scores based on document age.
    """
    for chunk in chunks:
        weight = compute_recency_weight(chunk, decay_days, min_weight)
        # Store the weight for potential use
        chunk["_recency_weight"] = weight

    return chunks


def build_cypher_filters(criteria: FilterCriteria) -> str:
    """
    Build Cypher WHERE clauses for Neo4j query filtering.

    Returns a string to be appended to WHERE clause.
    """
    conditions = []

    # Extension filter
    if criteria.allowed_extensions:
        exts = [f"'.{e.lower().lstrip('.')}'" for e in criteria.allowed_extensions]
        conditions.append(f"d.filename ENDS WITH ANY(ext IN [{', '.join(exts)}])")

    # Source pattern
    if criteria.source_pattern:
        conditions.append(f"d.path =~ '{criteria.source_pattern}'")

    if not conditions:
        return ""

    return " AND " + " AND ".join(conditions)
