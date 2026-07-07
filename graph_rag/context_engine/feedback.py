"""
Feedback collection and analytics for continuous improvement.

Implements:
- Explicit feedback logging (thumbs up/down, ratings)
- Implicit signal collection (click-through, dwell time)
- Retrieval quality metrics
- Analytics for prompt/retrieval tuning
"""

import os
import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Feedback storage configuration
FEEDBACK_DB_PATH = os.getenv("FEEDBACK_DB_PATH", ".feedback/feedback.db")
ANALYTICS_WINDOW_DAYS = int(os.getenv("ANALYTICS_WINDOW_DAYS", "30"))


class FeedbackType(Enum):
    """Types of feedback signals."""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"  # 1-5 scale
    CLICK = "click"
    NO_CLICK = "no_click"
    DWELL = "dwell"
    REFORMULATION = "reformulation"  # User rephrased query
    ESCALATION = "escalation"  # User escalated to human


@dataclass
class FeedbackEntry:
    """A single feedback entry."""
    feedback_id: str
    session_id: str
    user_id: Optional[str]
    query: str
    response: str
    feedback_type: FeedbackType
    value: float  # Normalized 0-1 for all types
    chunk_ids: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["feedback_type"] = self.feedback_type.value
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeedbackEntry":
        data = dict(data)
        if isinstance(data.get("feedback_type"), str):
            data["feedback_type"] = FeedbackType(data["feedback_type"])
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        if isinstance(data.get("chunk_ids"), str):
            data["chunk_ids"] = json.loads(data["chunk_ids"])
        if isinstance(data.get("metadata"), str):
            data["metadata"] = json.loads(data["metadata"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RetrievalMetrics:
    """Metrics for a retrieval operation."""
    query_id: str
    query: str
    intent: str
    num_chunks_retrieved: int
    num_chunks_used: int
    retrieval_time_ms: float
    rerank_time_ms: float = 0.0
    total_time_ms: float = 0.0
    diversity_score: float = 0.0
    avg_chunk_score: float = 0.0
    chunk_ids: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


class FeedbackStore:
    """SQLite-based feedback storage with analytics support."""

    _local = threading.local()

    def __init__(self, db_path: str = FEEDBACK_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
        return self._local.conn

    @contextmanager
    def _cursor(self):
        """Context manager for cursor with auto-commit."""
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._cursor() as cur:
            # Feedback table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_id TEXT,
                    query TEXT NOT NULL,
                    response TEXT,
                    feedback_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    chunk_ids TEXT,
                    timestamp TEXT NOT NULL,
                    metadata TEXT
                )
            """)

            # Retrieval metrics table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS retrieval_metrics (
                    query_id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    intent TEXT,
                    num_chunks_retrieved INTEGER,
                    num_chunks_used INTEGER,
                    retrieval_time_ms REAL,
                    rerank_time_ms REAL,
                    total_time_ms REAL,
                    diversity_score REAL,
                    avg_chunk_score REAL,
                    chunk_ids TEXT,
                    timestamp TEXT NOT NULL
                )
            """)

            # Indexes for analytics queries
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_timestamp
                ON feedback(timestamp)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_type
                ON feedback(feedback_type)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp
                ON retrieval_metrics(timestamp)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_intent
                ON retrieval_metrics(intent)
            """)

    def log_feedback(self, entry: FeedbackEntry) -> None:
        """Log a feedback entry."""
        with self._cursor() as cur:
            cur.execute("""
                INSERT OR REPLACE INTO feedback
                (feedback_id, session_id, user_id, query, response, feedback_type,
                 value, chunk_ids, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.feedback_id,
                entry.session_id,
                entry.user_id,
                entry.query,
                entry.response,
                entry.feedback_type.value,
                entry.value,
                json.dumps(entry.chunk_ids),
                entry.timestamp.isoformat(),
                json.dumps(entry.metadata)
            ))

    def log_metrics(self, metrics: RetrievalMetrics) -> None:
        """Log retrieval metrics."""
        with self._cursor() as cur:
            cur.execute("""
                INSERT OR REPLACE INTO retrieval_metrics
                (query_id, query, intent, num_chunks_retrieved, num_chunks_used,
                 retrieval_time_ms, rerank_time_ms, total_time_ms, diversity_score,
                 avg_chunk_score, chunk_ids, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.query_id,
                metrics.query,
                metrics.intent,
                metrics.num_chunks_retrieved,
                metrics.num_chunks_used,
                metrics.retrieval_time_ms,
                metrics.rerank_time_ms,
                metrics.total_time_ms,
                metrics.diversity_score,
                metrics.avg_chunk_score,
                json.dumps(metrics.chunk_ids),
                metrics.timestamp.isoformat()
            ))

    def get_feedback_stats(self, days: int = ANALYTICS_WINDOW_DAYS) -> Dict[str, Any]:
        """Get aggregated feedback statistics."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        with self._cursor() as cur:
            # Overall stats
            cur.execute("""
                SELECT
                    COUNT(*) as total,
                    AVG(value) as avg_value,
                    SUM(CASE WHEN feedback_type = 'thumbs_up' THEN 1 ELSE 0 END) as thumbs_up,
                    SUM(CASE WHEN feedback_type = 'thumbs_down' THEN 1 ELSE 0 END) as thumbs_down,
                    SUM(CASE WHEN feedback_type = 'rating' THEN 1 ELSE 0 END) as ratings,
                    AVG(CASE WHEN feedback_type = 'rating' THEN value ELSE NULL END) as avg_rating
                FROM feedback
                WHERE timestamp >= ?
            """, (cutoff,))
            row = cur.fetchone()

            stats = {
                "total_feedback": row["total"] or 0,
                "avg_value": row["avg_value"] or 0.0,
                "thumbs_up": row["thumbs_up"] or 0,
                "thumbs_down": row["thumbs_down"] or 0,
                "ratings_count": row["ratings"] or 0,
                "avg_rating": row["avg_rating"] or 0.0,
            }

            # Calculate satisfaction rate
            total_thumbs = stats["thumbs_up"] + stats["thumbs_down"]
            if total_thumbs > 0:
                stats["satisfaction_rate"] = stats["thumbs_up"] / total_thumbs
            else:
                stats["satisfaction_rate"] = None

            return stats

    def get_retrieval_stats(self, days: int = ANALYTICS_WINDOW_DAYS) -> Dict[str, Any]:
        """Get aggregated retrieval metrics."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        with self._cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) as total_queries,
                    AVG(retrieval_time_ms) as avg_retrieval_ms,
                    AVG(rerank_time_ms) as avg_rerank_ms,
                    AVG(total_time_ms) as avg_total_ms,
                    AVG(diversity_score) as avg_diversity,
                    AVG(avg_chunk_score) as avg_relevance,
                    AVG(num_chunks_retrieved) as avg_retrieved,
                    AVG(num_chunks_used) as avg_used
                FROM retrieval_metrics
                WHERE timestamp >= ?
            """, (cutoff,))
            row = cur.fetchone()

            return {
                "total_queries": row["total_queries"] or 0,
                "avg_retrieval_ms": row["avg_retrieval_ms"] or 0.0,
                "avg_rerank_ms": row["avg_rerank_ms"] or 0.0,
                "avg_total_ms": row["avg_total_ms"] or 0.0,
                "avg_diversity": row["avg_diversity"] or 0.0,
                "avg_relevance": row["avg_relevance"] or 0.0,
                "avg_chunks_retrieved": row["avg_retrieved"] or 0.0,
                "avg_chunks_used": row["avg_used"] or 0.0,
            }

    def get_intent_breakdown(self, days: int = ANALYTICS_WINDOW_DAYS) -> Dict[str, int]:
        """Get query counts by intent."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        with self._cursor() as cur:
            cur.execute("""
                SELECT intent, COUNT(*) as count
                FROM retrieval_metrics
                WHERE timestamp >= ? AND intent IS NOT NULL
                GROUP BY intent
                ORDER BY count DESC
            """, (cutoff,))

            return {row["intent"]: row["count"] for row in cur.fetchall()}

    def get_low_satisfaction_queries(self, days: int = ANALYTICS_WINDOW_DAYS,
                                      limit: int = 20) -> List[Dict[str, Any]]:
        """Get queries with low satisfaction for analysis."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        with self._cursor() as cur:
            cur.execute("""
                SELECT query, feedback_type, value, timestamp, metadata
                FROM feedback
                WHERE timestamp >= ?
                  AND (feedback_type = 'thumbs_down'
                       OR (feedback_type = 'rating' AND value < 0.4)
                       OR feedback_type = 'reformulation')
                ORDER BY timestamp DESC
                LIMIT ?
            """, (cutoff, limit))

            return [dict(row) for row in cur.fetchall()]

    def get_chunk_performance(self, days: int = ANALYTICS_WINDOW_DAYS,
                               limit: int = 50) -> List[Dict[str, Any]]:
        """Get chunk performance based on feedback."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        # This requires parsing chunk_ids JSON - simplified version
        with self._cursor() as cur:
            cur.execute("""
                SELECT chunk_ids, AVG(value) as avg_score, COUNT(*) as appearances
                FROM feedback
                WHERE timestamp >= ? AND chunk_ids IS NOT NULL AND chunk_ids != '[]'
                GROUP BY chunk_ids
                HAVING COUNT(*) >= 2
                ORDER BY avg_score DESC
                LIMIT ?
            """, (cutoff, limit))

            results = []
            for row in cur.fetchall():
                try:
                    chunk_ids = json.loads(row["chunk_ids"])
                    results.append({
                        "chunk_ids": chunk_ids,
                        "avg_score": row["avg_score"],
                        "appearances": row["appearances"]
                    })
                except Exception:
                    pass

            return results


class FeedbackCollector:
    """
    High-level API for collecting feedback.

    Usage:
        collector = FeedbackCollector()
        collector.thumbs_up(session_id, query, response, chunk_ids)
        collector.thumbs_down(session_id, query, response, chunk_ids, reason="irrelevant")

        # Get analytics
        stats = collector.get_analytics()
    """

    def __init__(self, store: Optional[FeedbackStore] = None):
        self.store = store or FeedbackStore()
        self._feedback_count = 0

    def _generate_id(self) -> str:
        """Generate unique feedback ID."""
        import uuid
        return str(uuid.uuid4())

    def thumbs_up(self, session_id: str, query: str, response: str,
                  chunk_ids: List[str] = None, user_id: str = None) -> str:
        """Record positive feedback."""
        entry = FeedbackEntry(
            feedback_id=self._generate_id(),
            session_id=session_id,
            user_id=user_id,
            query=query,
            response=response,
            feedback_type=FeedbackType.THUMBS_UP,
            value=1.0,
            chunk_ids=chunk_ids or []
        )
        self.store.log_feedback(entry)
        self._feedback_count += 1
        logger.debug(f"Logged thumbs_up for query: {query[:50]}...")
        return entry.feedback_id

    def thumbs_down(self, session_id: str, query: str, response: str,
                    chunk_ids: List[str] = None, user_id: str = None,
                    reason: str = None) -> str:
        """Record negative feedback."""
        entry = FeedbackEntry(
            feedback_id=self._generate_id(),
            session_id=session_id,
            user_id=user_id,
            query=query,
            response=response,
            feedback_type=FeedbackType.THUMBS_DOWN,
            value=0.0,
            chunk_ids=chunk_ids or [],
            metadata={"reason": reason} if reason else {}
        )
        self.store.log_feedback(entry)
        self._feedback_count += 1
        logger.debug(f"Logged thumbs_down for query: {query[:50]}...")
        return entry.feedback_id

    def rating(self, session_id: str, query: str, response: str,
               score: int, chunk_ids: List[str] = None,
               user_id: str = None) -> str:
        """Record 1-5 rating."""
        # Normalize to 0-1
        normalized = (score - 1) / 4.0

        entry = FeedbackEntry(
            feedback_id=self._generate_id(),
            session_id=session_id,
            user_id=user_id,
            query=query,
            response=response,
            feedback_type=FeedbackType.RATING,
            value=normalized,
            chunk_ids=chunk_ids or [],
            metadata={"raw_score": score}
        )
        self.store.log_feedback(entry)
        self._feedback_count += 1
        return entry.feedback_id

    def click(self, session_id: str, query: str, chunk_id: str,
              user_id: str = None) -> str:
        """Record click on a result."""
        entry = FeedbackEntry(
            feedback_id=self._generate_id(),
            session_id=session_id,
            user_id=user_id,
            query=query,
            response="",
            feedback_type=FeedbackType.CLICK,
            value=1.0,
            chunk_ids=[chunk_id]
        )
        self.store.log_feedback(entry)
        return entry.feedback_id

    def reformulation(self, session_id: str, original_query: str,
                      new_query: str, user_id: str = None) -> str:
        """Record query reformulation (implicit negative signal)."""
        entry = FeedbackEntry(
            feedback_id=self._generate_id(),
            session_id=session_id,
            user_id=user_id,
            query=original_query,
            response="",
            feedback_type=FeedbackType.REFORMULATION,
            value=0.3,  # Mild negative signal
            metadata={"new_query": new_query}
        )
        self.store.log_feedback(entry)
        return entry.feedback_id

    def log_retrieval(self, query_id: str, query: str, intent: str,
                      chunks: List[Dict[str, Any]],
                      retrieval_time_ms: float,
                      rerank_time_ms: float = 0.0,
                      diversity_score: float = 0.0) -> None:
        """Log retrieval metrics."""
        chunk_ids = [c.get("chunk_id", "") for c in chunks]
        scores = [c.get("_score", 0.0) for c in chunks]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        metrics = RetrievalMetrics(
            query_id=query_id,
            query=query,
            intent=intent,
            num_chunks_retrieved=len(chunks),
            num_chunks_used=min(len(chunks), 6),  # Typical top_k
            retrieval_time_ms=retrieval_time_ms,
            rerank_time_ms=rerank_time_ms,
            total_time_ms=retrieval_time_ms + rerank_time_ms,
            diversity_score=diversity_score,
            avg_chunk_score=avg_score,
            chunk_ids=chunk_ids
        )
        self.store.log_metrics(metrics)

    def get_analytics(self, days: int = ANALYTICS_WINDOW_DAYS) -> Dict[str, Any]:
        """Get comprehensive analytics."""
        return {
            "feedback": self.store.get_feedback_stats(days),
            "retrieval": self.store.get_retrieval_stats(days),
            "intent_breakdown": self.store.get_intent_breakdown(days),
            "period_days": days
        }

    def get_improvement_opportunities(self, days: int = ANALYTICS_WINDOW_DAYS) -> Dict[str, Any]:
        """Identify areas for improvement."""
        low_sat = self.store.get_low_satisfaction_queries(days)
        chunk_perf = self.store.get_chunk_performance(days)

        # Identify patterns in low-satisfaction queries
        patterns = {}
        for item in low_sat:
            query = item.get("query", "").lower()
            for keyword in ["how", "what", "when", "where", "compare", "list"]:
                if keyword in query:
                    patterns[keyword] = patterns.get(keyword, 0) + 1

        return {
            "low_satisfaction_queries": low_sat[:10],
            "query_patterns": patterns,
            "top_performing_chunks": [c for c in chunk_perf if c["avg_score"] > 0.7][:10],
            "low_performing_chunks": [c for c in chunk_perf if c["avg_score"] < 0.4][:10],
        }


# Global collector instance
_collector: Optional[FeedbackCollector] = None
_collector_lock = threading.Lock()


def get_feedback_collector() -> FeedbackCollector:
    """Get or create global feedback collector."""
    global _collector
    if _collector is None:
        with _collector_lock:
            if _collector is None:
                _collector = FeedbackCollector()
    return _collector
