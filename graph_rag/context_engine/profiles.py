"""
User profile management for personalized retrieval.

Implements:
- User preference tracking
- Query history analysis
- Personalized result boosting
- Role-based context adaptation
"""

import os
import json
import logging
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

# Profile configuration
PROFILE_TTL_DAYS = int(os.getenv("PROFILE_TTL_DAYS", "90"))
MAX_QUERY_HISTORY = int(os.getenv("MAX_QUERY_HISTORY", "100"))
MIN_TOPIC_FREQUENCY = int(os.getenv("MIN_TOPIC_FREQUENCY", "3"))


@dataclass
class UserPreferences:
    """User preferences for retrieval."""
    preferred_sources: List[str] = field(default_factory=list)
    blocked_sources: List[str] = field(default_factory=list)
    preferred_doc_types: List[str] = field(default_factory=list)
    language: str = "en"
    detail_level: str = "standard"  # brief, standard, detailed

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserPreferences":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class QueryHistoryEntry:
    """A single query history entry."""
    query: str
    timestamp: datetime
    intent: str
    topics: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    clicked_results: List[str] = field(default_factory=list)
    feedback_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryHistoryEntry":
        data = dict(data)
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class UserProfile:
    """
    User profile containing preferences and history.

    Used for personalized retrieval and result boosting.
    """
    user_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    preferences: UserPreferences = field(default_factory=UserPreferences)
    query_history: List[QueryHistoryEntry] = field(default_factory=list)
    topic_frequencies: Dict[str, int] = field(default_factory=dict)
    entity_frequencies: Dict[str, int] = field(default_factory=dict)
    role: str = "user"  # user, admin, analyst, etc.
    department: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_query(self, query: str, intent: str,
                  topics: List[str] = None,
                  entities: List[str] = None) -> None:
        """Record a query in history."""
        entry = QueryHistoryEntry(
            query=query,
            timestamp=datetime.now(),
            intent=intent,
            topics=topics or [],
            entities=entities or []
        )
        self.query_history.append(entry)

        # Trim history if needed
        if len(self.query_history) > MAX_QUERY_HISTORY:
            self.query_history = self.query_history[-MAX_QUERY_HISTORY:]

        # Update frequency counters
        for topic in (topics or []):
            self.topic_frequencies[topic] = self.topic_frequencies.get(topic, 0) + 1
        for entity in (entities or []):
            self.entity_frequencies[entity] = self.entity_frequencies.get(entity, 0) + 1

        self.last_active = datetime.now()

    def add_feedback(self, query: str, score: float,
                     clicked_results: List[str] = None) -> None:
        """Add feedback for the most recent matching query."""
        for entry in reversed(self.query_history):
            if entry.query == query:
                entry.feedback_score = score
                if clicked_results:
                    entry.clicked_results = clicked_results
                break

    def get_top_topics(self, n: int = 10) -> List[str]:
        """Get user's most frequent topics."""
        sorted_topics = sorted(
            self.topic_frequencies.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [t for t, count in sorted_topics[:n] if count >= MIN_TOPIC_FREQUENCY]

    def get_top_entities(self, n: int = 10) -> List[str]:
        """Get user's most frequently referenced entities."""
        sorted_entities = sorted(
            self.entity_frequencies.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [e for e, count in sorted_entities[:n] if count >= MIN_TOPIC_FREQUENCY]

    def get_recent_queries(self, n: int = 10) -> List[str]:
        """Get user's most recent queries."""
        return [e.query for e in self.query_history[-n:]]

    def get_successful_queries(self) -> List[QueryHistoryEntry]:
        """Get queries that received positive feedback."""
        return [e for e in self.query_history
                if e.feedback_score is not None and e.feedback_score >= 0.7]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize profile to dict."""
        return {
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "preferences": self.preferences.to_dict(),
            "query_history": [e.to_dict() for e in self.query_history],
            "topic_frequencies": self.topic_frequencies,
            "entity_frequencies": self.entity_frequencies,
            "role": self.role,
            "department": self.department,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        """Deserialize profile from dict."""
        data = dict(data)

        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("last_active"), str):
            data["last_active"] = datetime.fromisoformat(data["last_active"])

        if isinstance(data.get("preferences"), dict):
            data["preferences"] = UserPreferences.from_dict(data["preferences"])

        if isinstance(data.get("query_history"), list):
            data["query_history"] = [
                QueryHistoryEntry.from_dict(e) if isinstance(e, dict) else e
                for e in data["query_history"]
            ]

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ProfileStore:
    """Abstract base for profile storage."""

    def get(self, user_id: str) -> Optional[UserProfile]:
        raise NotImplementedError

    def save(self, profile: UserProfile) -> None:
        raise NotImplementedError

    def delete(self, user_id: str) -> None:
        raise NotImplementedError

    def list_users(self) -> List[str]:
        raise NotImplementedError


class InMemoryProfileStore(ProfileStore):
    """In-memory profile storage for development."""

    def __init__(self):
        self._profiles: Dict[str, UserProfile] = {}

    def get(self, user_id: str) -> Optional[UserProfile]:
        profile = self._profiles.get(user_id)
        if profile:
            # Check TTL
            age = datetime.now() - profile.last_active
            if age.days > PROFILE_TTL_DAYS:
                del self._profiles[user_id]
                return None
        return profile

    def save(self, profile: UserProfile) -> None:
        self._profiles[profile.user_id] = profile

    def delete(self, user_id: str) -> None:
        self._profiles.pop(user_id, None)

    def list_users(self) -> List[str]:
        return list(self._profiles.keys())


class FileProfileStore(ProfileStore):
    """File-based profile storage."""

    def __init__(self, storage_dir: str = ".profiles"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    def _get_path(self, user_id: str) -> str:
        # Hash user_id for filesystem safety
        safe_id = hashlib.sha256(user_id.encode()).hexdigest()[:16]
        return os.path.join(self.storage_dir, f"{safe_id}.json")

    def get(self, user_id: str) -> Optional[UserProfile]:
        path = self._get_path(user_id)
        if not os.path.exists(path):
            return None

        try:
            with open(path, 'r') as f:
                data = json.load(f)

            profile = UserProfile.from_dict(data)

            # Check TTL
            age = datetime.now() - profile.last_active
            if age.days > PROFILE_TTL_DAYS:
                os.remove(path)
                return None

            return profile
        except Exception as e:
            logger.warning(f"Failed to load profile for {user_id}: {e}")
            return None

    def save(self, profile: UserProfile) -> None:
        path = self._get_path(profile.user_id)
        try:
            with open(path, 'w') as f:
                json.dump(profile.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save profile for {profile.user_id}: {e}")

    def delete(self, user_id: str) -> None:
        path = self._get_path(user_id)
        if os.path.exists(path):
            os.remove(path)

    def list_users(self) -> List[str]:
        # Note: This returns hashed IDs, not original user_ids
        return [f[:-5] for f in os.listdir(self.storage_dir) if f.endswith('.json')]


class ProfileManager:
    """
    Manages user profiles for personalized retrieval.

    Usage:
        manager = ProfileManager()
        profile = manager.get_or_create("user123")
        profile.add_query("What is policy X?", "factual", topics=["policy"])
        manager.save(profile)
    """

    def __init__(self, store: Optional[ProfileStore] = None):
        self.store = store or InMemoryProfileStore()

    def get_or_create(self, user_id: str) -> UserProfile:
        """Get existing profile or create new one."""
        profile = self.store.get(user_id)
        if profile is None:
            profile = UserProfile(user_id=user_id)
            self.store.save(profile)
        return profile

    def save(self, profile: UserProfile) -> None:
        """Save profile changes."""
        self.store.save(profile)

    def delete(self, user_id: str) -> None:
        """Delete a user profile."""
        self.store.delete(user_id)

    def record_query(self, user_id: str, query: str, intent: str,
                     topics: List[str] = None,
                     entities: List[str] = None) -> UserProfile:
        """Record a query for a user."""
        profile = self.get_or_create(user_id)
        profile.add_query(query, intent, topics, entities)
        self.save(profile)
        return profile

    def record_feedback(self, user_id: str, query: str, score: float,
                        clicked_results: List[str] = None) -> None:
        """Record feedback for a query."""
        profile = self.store.get(user_id)
        if profile:
            profile.add_feedback(query, score, clicked_results)
            self.save(profile)


def compute_personalization_boost(chunk: Dict[str, Any],
                                   profile: UserProfile) -> float:
    """
    Compute a personalization boost score for a chunk.

    Based on:
    - Source preference matching
    - Topic relevance to user history
    - Entity relevance to user history

    Returns a multiplier (1.0 = no boost, >1.0 = positive boost).
    """
    boost = 1.0

    metadata = chunk.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}

    # Source preference boost
    source = (metadata.get("source", "") or chunk.get("source", "")).lower()
    path = (metadata.get("path", "") or chunk.get("path", "")).lower()

    for pref_source in profile.preferences.preferred_sources:
        if pref_source.lower() in source or pref_source.lower() in path:
            boost *= 1.2
            break

    for blocked_source in profile.preferences.blocked_sources:
        if blocked_source.lower() in source or blocked_source.lower() in path:
            boost *= 0.5
            break

    # Topic relevance boost
    chunk_text = (chunk.get("text", "") + " " + chunk.get("heading", "")).lower()
    user_topics = profile.get_top_topics(5)

    topic_matches = sum(1 for topic in user_topics if topic.lower() in chunk_text)
    if topic_matches > 0:
        boost *= (1 + 0.1 * min(topic_matches, 3))  # Max 30% boost

    # Entity relevance boost
    user_entities = profile.get_top_entities(5)
    entity_matches = sum(1 for entity in user_entities if entity.lower() in chunk_text)
    if entity_matches > 0:
        boost *= (1 + 0.1 * min(entity_matches, 3))  # Max 30% boost

    # Role-based boost (example: analysts get technical docs boosted)
    if profile.role == "analyst":
        if any(kw in chunk_text for kw in ["analysis", "data", "report", "metrics"]):
            boost *= 1.1
    elif profile.role == "admin":
        if any(kw in chunk_text for kw in ["policy", "procedure", "compliance", "regulation"]):
            boost *= 1.1

    return boost


def apply_personalization(chunks: List[Dict[str, Any]],
                          profile: Optional[UserProfile]) -> List[Dict[str, Any]]:
    """
    Apply personalization boosts to chunks.

    Modifies chunk scores based on user profile.
    """
    if not profile:
        return chunks

    for chunk in chunks:
        boost = compute_personalization_boost(chunk, profile)
        chunk["_personalization_boost"] = boost

        # Apply boost to existing score if present
        if "_score" in chunk:
            chunk["_score"] *= boost

    return chunks


def get_profile_context(profile: UserProfile) -> str:
    """
    Generate context string from user profile for prompt augmentation.

    This can be included in the system prompt to personalize responses.
    """
    parts = []

    if profile.role and profile.role != "user":
        parts.append(f"User role: {profile.role}")

    if profile.department:
        parts.append(f"Department: {profile.department}")

    top_topics = profile.get_top_topics(5)
    if top_topics:
        parts.append(f"User frequently asks about: {', '.join(top_topics)}")

    if profile.preferences.detail_level != "standard":
        parts.append(f"Preferred response style: {profile.preferences.detail_level}")

    return "\n".join(parts)
