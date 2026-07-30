"""
Session management for conversation context.

Provides in-memory session storage with optional Redis backend.
Sessions track conversation history, extracted entities, and user state.
"""

import os
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Optional
from abc import ABC, abstractmethod

from .types import SessionContext, ConversationTurn, Intent, utcnow

logger = logging.getLogger(__name__)

# Session TTL: sessions expire after this duration of inactivity
SESSION_TTL_MINUTES = int(os.getenv("SESSION_TTL_MINUTES", "60"))

# Maximum turns to keep in session history
MAX_SESSION_TURNS = int(os.getenv("MAX_SESSION_TURNS", "20"))

# Summarize turns that fall out of the window into session.metadata["summary"]
# so long conversations keep their earlier context. Costs one LLM call per
# trim; disabled by default.
SESSION_SUMMARY_ENABLED = os.getenv("SESSION_SUMMARY_ENABLED", "0") == "1"

# Run expired-session cleanup every N saves (the in-memory store leaks
# otherwise: get() only reaps the specific session being fetched)
CLEANUP_EVERY_N_SAVES = int(os.getenv("SESSION_CLEANUP_EVERY_N_SAVES", "50"))


class SessionStore(ABC):
    """Abstract base class for session storage backends."""

    @abstractmethod
    def get(self, session_id: str) -> Optional[SessionContext]:
        """Retrieve a session by ID."""
        pass

    @abstractmethod
    def set(self, session: SessionContext) -> None:
        """Store or update a session."""
        pass

    @abstractmethod
    def delete(self, session_id: str) -> None:
        """Delete a session."""
        pass

    @abstractmethod
    def cleanup_expired(self) -> int:
        """Remove expired sessions. Returns count of removed sessions."""
        pass


class InMemorySessionStore(SessionStore):
    """
    In-memory session storage with TTL.

    Thread-safe implementation suitable for single-process deployments.
    For distributed deployments, use RedisSessionStore.
    """

    def __init__(self, ttl_minutes: int = SESSION_TTL_MINUTES):
        self._sessions: Dict[str, SessionContext] = {}
        self._lock = threading.Lock()
        self._ttl = timedelta(minutes=ttl_minutes)

    def get(self, session_id: str) -> Optional[SessionContext]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None

            # Check expiration
            if utcnow() - session.last_active > self._ttl:
                del self._sessions[session_id]
                logger.debug(f"Session {session_id} expired")
                return None

            return session

    def set(self, session: SessionContext) -> None:
        with self._lock:
            # Trim turns if needed
            if len(session.turns) > MAX_SESSION_TURNS:
                session.turns = session.turns[-MAX_SESSION_TURNS:]

            self._sessions[session.session_id] = session

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def cleanup_expired(self) -> int:
        now = utcnow()
        expired = []

        with self._lock:
            for sid, session in self._sessions.items():
                if now - session.last_active > self._ttl:
                    expired.append(sid)

            for sid in expired:
                del self._sessions[sid]

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")

        return len(expired)


class RedisSessionStore(SessionStore):
    """
    Redis-backed session storage.

    Suitable for distributed deployments where multiple processes
    need to share session state.
    """

    def __init__(self, redis_url: Optional[str] = None, ttl_minutes: int = SESSION_TTL_MINUTES):
        self._ttl_seconds = ttl_minutes * 60
        self._redis = None
        self._prefix = "graph_rag:session:"

        redis_url = redis_url or os.getenv("REDIS_URL")
        if redis_url:
            try:
                import redis
                self._redis = redis.from_url(redis_url)
                logger.info("Connected to Redis for session storage")
            except ImportError:
                logger.warning("redis package not installed, falling back to in-memory")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}, falling back to in-memory")

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    def get(self, session_id: str) -> Optional[SessionContext]:
        if self._redis is None:
            return None

        try:
            data = self._redis.get(self._key(session_id))
            if data is None:
                return None

            obj = json.loads(data)
            session = SessionContext(
                session_id=obj["session_id"],
                entities=obj.get("entities", {}),
                topic_stack=obj.get("topic_stack", []),
                created_at=datetime.fromisoformat(obj["created_at"]),
                last_active=datetime.fromisoformat(obj["last_active"]),
                metadata=obj.get("metadata", {}),
            )

            # Rebuild turns
            for t in obj.get("turns", []):
                session.turns.append(ConversationTurn(
                    role=t["role"],
                    content=t["content"],
                    timestamp=datetime.fromisoformat(t["timestamp"]),
                    intent=Intent(t["intent"]) if t.get("intent") else None,
                    entities=t.get("entities", []),
                ))

            # Refresh TTL
            self._redis.expire(self._key(session_id), self._ttl_seconds)
            return session

        except Exception as e:
            logger.warning(f"Error reading session from Redis: {e}")
            return None

    def set(self, session: SessionContext) -> None:
        if self._redis is None:
            return

        try:
            # Trim turns
            if len(session.turns) > MAX_SESSION_TURNS:
                session.turns = session.turns[-MAX_SESSION_TURNS:]

            data = json.dumps(session.to_dict())
            self._redis.setex(self._key(session.session_id), self._ttl_seconds, data)

        except Exception as e:
            logger.warning(f"Error writing session to Redis: {e}")

    def delete(self, session_id: str) -> None:
        if self._redis is None:
            return

        try:
            self._redis.delete(self._key(session_id))
        except Exception as e:
            logger.warning(f"Error deleting session from Redis: {e}")

    def cleanup_expired(self) -> int:
        # Redis handles expiration automatically via TTL
        return 0


class SessionManager:
    """
    High-level session management interface.

    Abstracts the storage backend and provides convenient methods
    for session manipulation.
    """

    def __init__(self, store: Optional[SessionStore] = None):
        self._save_count = 0
        if store is not None:
            self._store = store
        elif os.getenv("REDIS_URL"):
            # Try Redis first
            redis_store = RedisSessionStore()
            if redis_store._redis is not None:
                self._store = redis_store
            else:
                self._store = InMemorySessionStore()
        else:
            self._store = InMemorySessionStore()

        logger.info(f"Session manager initialized with {type(self._store).__name__}")

    def get_or_create(self, session_id: str) -> SessionContext:
        """Get existing session or create a new one."""
        session = self._store.get(session_id)
        if session is None:
            session = SessionContext(session_id=session_id)
            self._store.set(session)
            logger.debug(f"Created new session: {session_id}")
        return session

    def get(self, session_id: str) -> Optional[SessionContext]:
        """Get session if it exists."""
        return self._store.get(session_id)

    def save(self, session: SessionContext) -> None:
        """Save session state."""
        session.last_active = utcnow()

        # Fold turns that are about to fall out of the window into a rolling
        # summary so long conversations keep their earlier context.
        if SESSION_SUMMARY_ENABLED and len(session.turns) > MAX_SESSION_TURNS:
            self._summarize_dropped_turns(session)

        self._store.set(session)

        # Opportunistic reaping of expired sessions
        self._save_count += 1
        if CLEANUP_EVERY_N_SAVES > 0 and self._save_count % CLEANUP_EVERY_N_SAVES == 0:
            self.cleanup()

    def _summarize_dropped_turns(self, session: SessionContext) -> None:
        """Summarize the turns that will be trimmed into metadata['summary'].

        Best-effort: on any failure the turns are simply dropped as before.
        """
        dropped = session.turns[:-MAX_SESSION_TURNS]
        if not dropped:
            return

        try:
            from ..config import chat_client, CHAT_MODEL

            convo = "\n".join(
                f"{'User' if t.role == 'user' else 'Assistant'}: {t.content[:300]}"
                for t in dropped
            )
            prior = session.metadata.get("summary", "")
            prompt = (
                "Summarize the following conversation fragment in 2-3 sentences, "
                "keeping named entities and topics. "
                + (f"Fold in this earlier summary: {prior}\n\n" if prior else "\n")
                + convo
            )
            resp = chat_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=150,
            )
            summary = (resp.choices[0].message.content or "").strip()
            if summary:
                session.metadata["summary"] = summary[:1000]
                logger.debug(f"Rolled {len(dropped)} turns into session summary")
        except Exception as e:
            logger.debug(f"Session summarization failed (turns dropped): {e}")

    def delete(self, session_id: str) -> None:
        """Delete a session."""
        self._store.delete(session_id)

    def add_user_turn(self, session_id: str, content: str,
                      intent: Optional[Intent] = None,
                      entities: Optional[list] = None) -> SessionContext:
        """Add a user turn to the session."""
        session = self.get_or_create(session_id)
        session.add_turn("user", content, intent=intent, entities=entities)
        self.save(session)
        return session

    def add_assistant_turn(self, session_id: str, content: str) -> SessionContext:
        """Add an assistant turn to the session."""
        session = self.get_or_create(session_id)
        session.add_turn("assistant", content)
        self.save(session)
        return session

    def get_conversation_context(self, session_id: str, n_turns: int = 5) -> str:
        """Get recent conversation as text for LLM context."""
        session = self.get(session_id)
        if session is None:
            return ""
        return session.get_conversation_text(n_turns)

    def cleanup(self) -> int:
        """Clean up expired sessions."""
        return self._store.cleanup_expired()


# Global session manager instance
_session_manager: Optional[SessionManager] = None
_manager_lock = threading.Lock()


def get_session_manager() -> SessionManager:
    """Get or create the global session manager."""
    global _session_manager
    if _session_manager is None:
        with _manager_lock:
            if _session_manager is None:
                _session_manager = SessionManager()
    return _session_manager
