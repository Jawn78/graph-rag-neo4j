"""
Graph query operations for hybrid search.

Implements:
- Vector similarity search using Neo4j vector index
- Full-text keyword search using Lucene index
- Reciprocal Rank Fusion (RRF) for combining results

Data is trusted to be already sanitized during ingestion.
No re-sanitization is performed on query results.
"""

import logging
from typing import Any, Dict, List, Tuple

from ..config import CHUNK_INDEX_NAME, NEO4J_DB
from ..utils.logging import get_trace_id

logger = logging.getLogger(__name__)

# RRF parameter: dampens rank differences
# Higher values give more weight to lower-ranked results
# 60 is a commonly used value in literature (Cormack et al., 2009)
RRF_K_DEFAULT = 60


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
        with driver.session(database=NEO4J_DB) as s:
            rows = s.run("""
                CALL db.index.vector.queryNodes($index, $k, $vec)
                YIELD node, score
                MATCH (d:Document {doc_id: node.doc_id})
                RETURN node.chunk_id AS chunk_id, node.text AS text, node.heading AS heading,
                       node.`order` AS `order`, node.doc_id AS doc_id, d.title AS title, score
                ORDER BY score DESC
            """, index=CHUNK_INDEX_NAME, k=top_k, vec=q_vec).data()

    except Exception as e:
        logger.warning(f"[{trace_id}] Vector search failed: {e}")
        return []

    # Data was sanitized during ingestion - trust it
    results = []
    for r in rows or []:
        chunk = {
            "chunk_id": r["chunk_id"],
            "text": r["text"] or "",
            "heading": r.get("heading") or "",
            "order": r["order"],
            "doc_id": r["doc_id"],
            "title": r["title"] or ""
        }
        results.append((chunk, float(r["score"])))

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
        with driver.session(database=NEO4J_DB) as s:
            rows = s.run("""
                CALL db.index.fulltext.queryNodes('chunk_text_fts', $q, {limit: $k})
                YIELD node, score
                MATCH (d:Document {doc_id: node.doc_id})
                RETURN node.chunk_id AS chunk_id, node.text AS text, node.heading AS heading,
                       node.`order` AS `order`, node.doc_id AS doc_id, d.title AS title, score
                ORDER BY score DESC
            """, q=q_text, k=top_k).data()

    except Exception as e:
        logger.warning(f"[{trace_id}] Keyword search failed: {e}")
        return []

    # Data was sanitized during ingestion - trust it
    results = []
    for r in rows or []:
        chunk = {
            "chunk_id": r["chunk_id"],
            "text": r["text"] or "",
            "heading": r.get("heading") or "",
            "order": r["order"],
            "doc_id": r["doc_id"],
            "title": r["title"] or ""
        }
        results.append((chunk, float(r["score"])))

    logger.debug(f"[{trace_id}] Keyword search returned {len(results)} results")
    return results


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
        List of chunk dicts sorted by RRF score descending
    """
    trace_id = get_trace_id()

    try:
        # Fetch 2x top_k from each source for better fusion
        # (reduced from 3x/4x to minimize over-fetching)
        v_results = vector_search(driver, q_vec, top_k=top_k * 2)
        k_results = keyword_search(driver, q_text, top_k=top_k * 2)
    except Exception as e:
        logger.warning(f"[{trace_id}] Hybrid search failed: {e}")
        return []

    def build_rank_maps(pairs: List[Tuple[Dict[str, Any], float]]):
        """Build chunk_id -> rank and chunk_id -> record mappings."""
        # Sort by score descending
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

    # Union of all chunk IDs
    all_ids = set(v_rank) | set(k_rank)
    if not all_ids:
        logger.debug(f"[{trace_id}] Hybrid search: no results from either source")
        return []

    # Compute RRF scores
    rrf_scores = {}
    for cid in all_ids:
        score = 0.0
        if cid in v_rank:
            score += 1.0 / (rrf_k + v_rank[cid])
        if cid in k_rank:
            score += 1.0 / (rrf_k + k_rank[cid])
        rrf_scores[cid] = score

    # Sort by RRF score and take top_k
    ranked_ids = sorted(all_ids, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

    # Build output (prefer vector result if available, else keyword)
    output = []
    for cid in ranked_ids:
        output.append(v_rec.get(cid) or k_rec[cid])

    logger.debug(f"[{trace_id}] Hybrid search: {len(v_results)} vector + {len(k_results)} keyword -> {len(output)} fused")
    return output
