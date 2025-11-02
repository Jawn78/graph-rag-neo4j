"""
Batched upserts for Documents and Chunks.

- Accepts input rows in either of these shapes:
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

- For Documents, we also persist top-level properties:
    doc.source, doc.path, doc.filename
  (extracted from metadata if present)
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List

from neo4j import Driver
from ..config import NEO4J_DB

def _to_json(meta: Any) -> str:
    if not meta:
        return "{}"
    if isinstance(meta, str):
        return meta
    try:
        return json.dumps(meta, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return "{}"

def _extract_meta_fields(meta_any: Any) -> Dict[str, str]:
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
    out: List[Dict[str, Any]] = []
    for d in rows or []:
        meta_any = d.get("meta") if "meta" in d else d.get("metadata", {})
        meta_json = _to_json(meta_any)
        tops = _extract_meta_fields(meta_any)
        
        # Sanitize document text
        title = _sanitize_text(d.get("title", ""))
        text = _sanitize_text(d.get("text", ""))
        
        # Skip documents with corrupted text
        if not text or _is_corrupted_text(text):
            print(f"[WARNING] Skipping corrupted document: {d.get('doc_id', 'unknown')[:8]}...")
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

def _sanitize_text(text: str) -> str:
    """Sanitize text for database storage."""
    if not text:
        return ""
    
    import re
    
    # Remove control characters (except \t, \n, \r)
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', ' ', text)
    
    # Remove problematic Unicode characters
    text = re.sub(r'[\u200b-\u200d\ufeff]', '', text)  # Zero-width characters
    text = re.sub(r'[\u2028\u2029]', '\n', text)       # Line/paragraph separators
    text = re.sub(r'[\u00a0]', ' ', text)              # Non-breaking space
    
    # Remove excessive repeated characters
    text = re.sub(r'(.)\1{10,}', r'\1', text)
    
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    
    # Ensure proper encoding
    try:
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
    except Exception:
        pass
    
    return text.strip()

def _is_corrupted_text(text: str) -> bool:
    """Check if text appears to be corrupted."""
    if not text or len(text) < 10:
        return False
    
    import re
    
    # Check for excessive repeated characters
    if re.search(r'(.)\1{20,}', text):
        return True
    
    # Check for high ratio of non-printable characters
    printable_chars = sum(1 for c in text if c.isprintable() or c.isspace())
    if len(text) > 0 and printable_chars / len(text) < 0.7:
        return True
    
    return False

def _normalize_chunk_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for c in rows or []:
        meta_any = c.get("meta") if "meta" in c else c.get("metadata", {})
        
        # Sanitize text before storing
        text = c.get("text", "")
        heading = c.get("heading", "")
        
        # Apply aggressive text cleaning
        text = _sanitize_text(text)
        heading = _sanitize_text(heading)
        
        # Skip chunks with corrupted or empty text
        if not text or _is_corrupted_text(text):
            print(f"[WARNING] Skipping corrupted chunk: {c.get('chunk_id', 'unknown')[:8]}...")
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

def upsert_docs(dr: Driver, rows: Iterable[Dict[str, Any]]) -> None:
    norm = _normalize_doc_rows(rows)
    if not norm:
        return
    with dr.session(database=NEO4J_DB) as s:
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

def upsert_chunks(dr: Driver, rows: Iterable[Dict[str, Any]]) -> None:
    norm = _normalize_chunk_rows(rows)
    if not norm:
        return
    with dr.session(database=NEO4J_DB) as s:
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
