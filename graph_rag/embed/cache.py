"""
Embedding cache with SQLite backend.

Optimizations:
- WAL mode for concurrent read/write
- Batch lookups to reduce query count
- Connection pooling via thread-local storage
- Pre-compiled statements where possible
"""

import sqlite3
import json
import os
import threading
from typing import Dict, List, Optional

from ..config import EMBED_MODEL

DB_PATH = os.getenv("EMBED_CACHE_DB", "embed_cache.sqlite")

# Thread-local storage for connections
_local = threading.local()
_init_lock = threading.Lock()
_initialized = False


def _get_connection() -> sqlite3.Connection:
    """Get a thread-local database connection."""
    if not hasattr(_local, 'conn') or _local.conn is None:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        # Enable WAL mode for better concurrent access
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")  # ~40MB cache
        _local.conn = conn
    return _local.conn


def _ensure_schema() -> None:
    """Ensure the cache table exists. Called once per process."""
    global _initialized
    if _initialized:
        return

    with _init_lock:
        if _initialized:
            return

        conn = _get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                chunk_id TEXT NOT NULL,
                model    TEXT NOT NULL,
                dim      INTEGER NOT NULL,
                vec_json TEXT NOT NULL,
                PRIMARY KEY (chunk_id, model)
            )
        """)
        # Index for faster lookups
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_chunk_model
            ON cache(chunk_id, model)
        """)
        conn.commit()
        _initialized = True


def get(chunk_id: str) -> Optional[List[float]]:
    """
    Get a single embedding from cache.

    For bulk operations, prefer get_batch() which is much faster.

    Args:
        chunk_id: The chunk ID to look up

    Returns:
        The embedding vector, or None if not cached
    """
    _ensure_schema()
    conn = _get_connection()
    row = conn.execute(
        "SELECT vec_json FROM cache WHERE chunk_id=? AND model=?",
        (chunk_id, EMBED_MODEL)
    ).fetchone()
    return json.loads(row[0]) if row else None


def get_batch(chunk_ids: List[str]) -> Dict[str, List[float]]:
    """
    Get multiple embeddings from cache in a single query.

    This is 50-100x faster than calling get() in a loop.

    Args:
        chunk_ids: List of chunk IDs to look up

    Returns:
        Dict mapping chunk_id -> embedding vector (only for found entries)
    """
    if not chunk_ids:
        return {}

    _ensure_schema()
    conn = _get_connection()

    # SQLite has a limit on the number of variables in a query (default 999)
    # Process in batches if needed
    result: Dict[str, List[float]] = {}
    batch_size = 500

    for i in range(0, len(chunk_ids), batch_size):
        batch = chunk_ids[i:i + batch_size]
        placeholders = ','.join('?' * len(batch))
        params = batch + [EMBED_MODEL]

        cursor = conn.execute(
            f"SELECT chunk_id, vec_json FROM cache WHERE chunk_id IN ({placeholders}) AND model=?",
            params
        )
        for row in cursor:
            result[row[0]] = json.loads(row[1])

    return result


def put(chunk_id: str, vec: List[float]) -> None:
    """
    Store a single embedding in cache.

    For bulk operations, prefer put_batch() which is much faster.

    Args:
        chunk_id: The chunk ID
        vec: The embedding vector
    """
    _ensure_schema()
    conn = _get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO cache(chunk_id, model, dim, vec_json) VALUES(?,?,?,?)",
        (chunk_id, EMBED_MODEL, len(vec), json.dumps(vec, separators=(',', ':')))
    )
    conn.commit()


def put_batch(embeddings: Dict[str, List[float]]) -> None:
    """
    Store multiple embeddings in cache in a single transaction.

    This is much faster than calling put() in a loop.

    Args:
        embeddings: Dict mapping chunk_id -> embedding vector
    """
    if not embeddings:
        return

    _ensure_schema()
    conn = _get_connection()

    # Use executemany for batch insert
    rows = [
        (chunk_id, EMBED_MODEL, len(vec), json.dumps(vec, separators=(',', ':')))
        for chunk_id, vec in embeddings.items()
    ]

    conn.executemany(
        "INSERT OR REPLACE INTO cache(chunk_id, model, dim, vec_json) VALUES(?,?,?,?)",
        rows
    )
    conn.commit()


def clear() -> None:
    """Clear all cached embeddings."""
    _ensure_schema()
    conn = _get_connection()
    conn.execute("DELETE FROM cache")
    conn.commit()


def close() -> None:
    """Close the thread-local connection."""
    if hasattr(_local, 'conn') and _local.conn is not None:
        _local.conn.close()
        _local.conn = None
