"""
Batched upserts for Documents and Chunks.

Accepts input rows in either of these shapes:
    Docs:
      {"doc_id": str, "title": str, "text": str,
       "meta": "<json string>" }  OR
      {"doc_id": str, "title": str, "text": str,
       "metadata": { ... } }

    Chunks:
      {"chunk_id": str, "doc_id": str, "order": int, "text": str, "heading": str,
       "embedding": [float,...],
       "meta": "<json string>"} OR
      {"chunk_id": str, "doc_id": str, "order": int, "text": str, "heading": str,
       "embedding": [float,...],
       "metadata": { ... } }

For Documents, we also persist top-level properties:
    doc.source, doc.path, doc.filename
  (extracted from metadata if present)

Data is expected to be already sanitized from the ingestion pipeline.
This module performs only minimal validation to catch corruption.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List

from neo4j import Driver

from ..config import NEO4J_DB
from ..utils.text import sanitize_text, is_corrupted
from ..utils.logging import get_trace_id

logger = logging.getLogger(__name__)


def _to_json(meta: Any) -> str:
    """Convert metadata to JSON string."""
    if not meta:
        return "{}"
    if isinstance(meta, str):
        return meta
    try:
        return json.dumps(meta, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return "{}"


def _extract_meta_fields(meta_any: Any) -> Dict[str, str]:
    """Extract common metadata fields from various input formats."""
    obj: Dict[str, Any] = {}
    if isinstance(meta_any, dict):
        obj = meta_any
    elif isinstance(meta_any, str) and meta_any.strip():
        try:
            obj = json.loads(meta_any)
        except Exception:
            obj = {}
    return {
        "source": str(obj.get("source", "") or ""),
        "path": str(obj.get("path", "") or ""),
        "filename": str(obj.get("filename", "") or ""),
    }


def _normalize_doc_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize document rows for upsert.

    Performs validation and light sanitization. Data should already be
    sanitized from ingestion, but we catch any corruption here.
    """
    trace_id = get_trace_id()
    out: List[Dict[str, Any]] = []

    for d in rows or []:
        meta_any = d.get("meta") if "meta" in d else d.get("metadata", {})
        meta_json = _to_json(meta_any)
        tops = _extract_meta_fields(meta_any)

        # Light sanitization (should already be clean from ingestion)
        title = sanitize_text(d.get("title", ""), check_corruption=False)
        text = d.get("text", "") or ""

        # Validation: skip corrupted documents
        if not text or is_corrupted(text):
            logger.warning(f"[{trace_id}] Skipping corrupted document: {d.get('doc_id', 'unknown')[:8]}...")
            continue

        out.append({
            "doc_id": d["doc_id"],
            "title": title,
            "text": text,
            "meta_json": meta_json,
            "source": tops["source"],
            "path": tops["path"],
            "filename": tops["filename"],
        })

    return out


def _normalize_chunk_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize chunk rows for upsert.

    Performs validation. Data should already be sanitized from ingestion.
    """
    trace_id = get_trace_id()
    out: List[Dict[str, Any]] = []

    for c in rows or []:
        meta_any = c.get("meta") if "meta" in c else c.get("metadata", {})
        text = c.get("text", "") or ""
        heading = c.get("heading", "") or ""

        # Validation: skip corrupted chunks
        if not text or is_corrupted(text):
            logger.warning(f"[{trace_id}] Skipping corrupted chunk: {c.get('chunk_id', 'unknown')[:8]}...")
            continue

        out.append({
            "chunk_id": c["chunk_id"],
            "doc_id": c["doc_id"],
            "order": int(c.get("order", 0)),
            "text": text,
            "heading": heading,
            "meta_json": _to_json(meta_any),
            "embedding": c.get("embedding"),
        })

    return out


def upsert_docs(driver: Driver, rows: Iterable[Dict[str, Any]]) -> int:
    """
    Upsert documents to Neo4j.

    Args:
        driver: Neo4j driver
        rows: Iterable of document dicts

    Returns:
        Number of documents upserted
    """
    trace_id = get_trace_id()
    norm = _normalize_doc_rows(rows)

    if not norm:
        return 0

    with driver.session(database=NEO4J_DB) as s:
        s.run(
            """
            UNWIND $rows AS d
            MERGE (doc:Document {doc_id:d.doc_id})
            ON CREATE SET
              doc.title         = d.title,
              doc.text          = d.text,
              doc.metadata_json = d.meta_json,
              doc.source        = d.source,
              doc.path          = d.path,
              doc.filename      = d.filename
            ON MATCH SET
              doc.title         = d.title,
              doc.metadata_json = d.meta_json,
              doc.source        = d.source,
              doc.path          = d.path,
              doc.filename      = d.filename
            """,
            rows=norm,
        )

    logger.debug(f"[{trace_id}] Upserted {len(norm)} documents")
    return len(norm)


def upsert_chunks(driver: Driver, rows: Iterable[Dict[str, Any]]) -> int:
    """
    Upsert chunks to Neo4j with embeddings.

    Args:
        driver: Neo4j driver
        rows: Iterable of chunk dicts

    Returns:
        Number of chunks upserted
    """
    trace_id = get_trace_id()
    norm = _normalize_chunk_rows(rows)

    if not norm:
        return 0

    with driver.session(database=NEO4J_DB) as s:
        s.run(
            """
            UNWIND $rows AS c
            MERGE (ch:Chunk {chunk_id:c.chunk_id})
            ON CREATE SET
              ch.text          = c.text,
              ch.heading       = c.heading,
              ch.`order`       = c.`order`,
              ch.doc_id        = c.doc_id,
              ch.metadata_json = c.meta_json,
              ch.embedding     = c.embedding
            ON MATCH SET
              ch.text          = c.text,
              ch.heading       = c.heading,
              ch.`order`       = c.`order`,
              ch.doc_id        = c.doc_id,
              ch.metadata_json = c.meta_json,
              ch.embedding     = c.embedding
            WITH ch, c
            MATCH (d:Document {doc_id:c.doc_id})
            MERGE (d)-[:HAS_CHUNK]->(ch)
            """,
            rows=norm,
        )

    logger.debug(f"[{trace_id}] Upserted {len(norm)} chunks")
    return len(norm)
