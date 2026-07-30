"""
Graph query operations for hybrid search.

Implements:
- Vector similarity search using Neo4j vector index
- Full-text keyword search using Lucene index
- Reciprocal Rank Fusion (RRF) for combining results

Data is trusted to be already sanitized during ingestion.
No re-sanitization is performed on query results.

Every returned chunk dict carries:
- "_score": the retrieval score (raw index score, or RRF score from
  hybrid_search). Downstream consumers (reranker fallback, personalization,
  feedback analytics) rely on this key being present.
- "metadata": a dict with source/path/filename/ext from the parent Document,
  used by context_engine.filters.
"""

import os
import logging
from typing import Any, Dict, List, Tuple

from neo4j import READ_ACCESS

from ..config import CHUNK_INDEX_NAME, NEO4J_DB
from ..utils.logging import get_trace_id

logger = logging.getLogger(__name__)

# RRF parameter: dampens rank differences
# Higher values give more weight to lower-ranked results
# 60 is a commonly used value in literature (Cormack et al., 2009)
RRF_K_DEFAULT = 60

_CHUNK_RETURN = """
    RETURN node.chunk_id AS chunk_id, node.text AS text, node.heading AS heading,
           node.`order` AS `order`, node.doc_id AS doc_id, d.title AS title,
           d.source AS source, d.path AS path, d.filename AS filename, score
    ORDER BY score DESC
"""


def _row_to_chunk(r: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a query result row into the canonical chunk dict."""
    filename = r.get("filename") or ""
    return {
        "chunk_id": r["chunk_id"],
        "text": r["text"] or "",
        "heading": r.get("heading") or "",
        "order": r["order"],
        "doc_id": r["doc_id"],
        "title": r["title"] or "",
        "_score": float(r["score"]),
        "metadata": {
            "source": r.get("source") or "",
            "path": r.get("path") or "",
            "filename": filename,
            "ext": os.path.splitext(filename)[1].lower(),
        },
    }


def vector_search(driver, q_vec: List[float], top_k: int = 24) -> List[Tuple[Dict[str, Any], float]]:
    """
    Search for chunks by vector similarity.

    Args:
        driver: Neo4j driver
        q_vec: Query embedding vector
        top_k: Number of results to return

    Returns:
        List of (chunk_dict, score) tuples sorted by score descending
    """
    trace_id = get_trace_id()

    try:
        with driver.session(database=NEO4J_DB, default_access_mode=READ_ACCESS) as s:
            rows = s.run("""
                CALL db.index.vector.queryNodes($index, $k, $vec)
                YIELD node, score
                MATCH (d:Document {doc_id: node.doc_id})
            """ + _CHUNK_RETURN, index=CHUNK_INDEX_NAME, k=top_k, vec=q_vec).data()

    except Exception as e:
        logger.warning(f"[{trace_id}] Vector search failed: {e}")
        return []

    results = [(chunk, chunk["_score"]) for chunk in map(_row_to_chunk, rows or [])]
    logger.debug(f"[{trace_id}] Vector search returned {len(results)} results")
    return results


def keyword_search(driver, q_text: str, top_k: int = 32) -> List[Tuple[Dict[str, Any], float]]:
    """
    Search for chunks by keyword matching.

    Args:
        driver: Neo4j driver
        q_text: Query text for full-text search
        top_k: Number of results to return

    Returns:
        List of (chunk_dict, score) tuples sorted by score descending
    """
    trace_id = get_trace_id()

    try:
        with driver.session(database=NEO4J_DB, default_access_mode=READ_ACCESS) as s:
            rows = s.run("""
                CALL db.index.fulltext.queryNodes('chunk_text_fts', $q, {limit: $k})
                YIELD node, score
                MATCH (d:Document {doc_id: node.doc_id})
            """ + _CHUNK_RETURN, q=q_text, k=top_k).data()

    except Exception as e:
        logger.warning(f"[{trace_id}] Keyword search failed: {e}")
        return []

    results = [(chunk, chunk["_score"]) for chunk in map(_row_to_chunk, rows or [])]
    logger.debug(f"[{trace_id}] Keyword search returned {len(results)} results")
    return results


def rrf_fuse(v_results: List[Tuple[Dict[str, Any], float]],
             k_results: List[Tuple[Dict[str, Any], float]],
             top_k: int, rrf_k: int = RRF_K_DEFAULT) -> List[Dict[str, Any]]:
    """
    Fuse two ranked result lists with Reciprocal Rank Fusion.

    Pure function (no I/O) so it can be unit-tested. The fused RRF score is
    stored on each returned chunk as "_score".
    """
    def build_rank_maps(pairs: List[Tuple[Dict[str, Any], float]]):
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        id_to_rank = {}
        id_to_record = {}
        for i, (rec, _) in enumerate(sorted_pairs, start=1):
            cid = rec["chunk_id"]
            id_to_rank[cid] = i
            id_to_record[cid] = rec
        return id_to_rank, id_to_record

    v_rank, v_rec = build_rank_maps(v_results)
    k_rank, k_rec = build_rank_maps(k_results)

    all_ids = set(v_rank) | set(k_rank)
    if not all_ids:
        return []

    rrf_scores = {}
    for cid in all_ids:
        score = 0.0
        if cid in v_rank:
            score += 1.0 / (rrf_k + v_rank[cid])
        if cid in k_rank:
            score += 1.0 / (rrf_k + k_rank[cid])
        rrf_scores[cid] = score

    ranked_ids = sorted(all_ids, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

    output = []
    for cid in ranked_ids:
        rec = v_rec.get(cid) or k_rec[cid]
        rec["_score"] = rrf_scores[cid]
        output.append(rec)
    return output


def hybrid_search(driver, q_vec: List[float], q_text: str, top_k: int = 12,
                  rrf_k: int = RRF_K_DEFAULT) -> List[Dict[str, Any]]:
    """
    Combine vector and keyword search using Reciprocal Rank Fusion.

    RRF score formula:
        score(doc) = sum(1 / (rrf_k + rank_i(doc))) for each ranker i

    This gives balanced weight to both vector (semantic) and keyword (lexical)
    matches without requiring score normalization.

    Args:
        driver: Neo4j driver
        q_vec: Query embedding vector
        q_text: Query text for keyword search
        top_k: Number of results to return
        rrf_k: RRF smoothing parameter (default 60)

    Returns:
        List of chunk dicts sorted by RRF score descending, with the RRF
        score attached as "_score"
    """
    trace_id = get_trace_id()

    # Fetch 2x top_k from each source for better fusion
    v_results = vector_search(driver, q_vec, top_k=top_k * 2)
    k_results = keyword_search(driver, q_text, top_k=top_k * 2)

    output = rrf_fuse(v_results, k_results, top_k=top_k, rrf_k=rrf_k)

    logger.debug(f"[{trace_id}] Hybrid search: {len(v_results)} vector + {len(k_results)} keyword -> {len(output)} fused")
    return output
