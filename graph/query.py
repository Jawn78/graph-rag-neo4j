# graph_rag/graph/query.py
from typing import Any, Dict, List, Tuple
from ..config import CHUNK_INDEX_NAME, NEO4J_DB

def vector_search(driver, q_vec: List[float], top_k: int = 24) -> List[Tuple[Dict[str, Any], float]]:
    try:
        with driver.session(database=NEO4J_DB) as s:
            # Optimized query with better performance
            rows = s.run("""
                CALL db.index.vector.queryNodes($index, $k, $vec)
                YIELD node, score
                MATCH (d:Document {doc_id: node.doc_id})
                RETURN node.chunk_id AS chunk_id, node.text AS text, node.heading AS heading,
                       node.`order` AS `order`, node.doc_id AS doc_id, d.title AS title, score
                ORDER BY score DESC
            """, index=CHUNK_INDEX_NAME, k=top_k, vec=q_vec).data()
    except Exception:
        return []
    
    # Sanitize results before returning
    def sanitize_chunk_data(data):
        from .upsert import _sanitize_text
        return {
            "chunk_id": data["chunk_id"],
            "text": _sanitize_text(data["text"]),
            "heading": _sanitize_text(data.get("heading", "")),
            "order": data["order"],
            "doc_id": data["doc_id"],
            "title": _sanitize_text(data["title"] or "")
        }
    
    return [(sanitize_chunk_data(r), float(r["score"])) for r in rows or []]

def keyword_search(driver, q_text: str, top_k: int = 32) -> List[Tuple[Dict[str, Any], float]]:
    try:
        with driver.session(database=NEO4J_DB) as s:
            # Optimized query with better performance
            rows = s.run("""
                CALL db.index.fulltext.queryNodes('chunk_text_fts', $q, {limit: $k})
                YIELD node, score
                MATCH (d:Document {doc_id: node.doc_id})
                RETURN node.chunk_id AS chunk_id, node.text AS text, node.heading AS heading,
                       node.`order` AS `order`, node.doc_id AS doc_id, d.title AS title, score
                ORDER BY score DESC
            """, q=q_text, k=top_k).data()
    except Exception:
        return []
    
    # Sanitize results before returning
    def sanitize_chunk_data(data):
        from .upsert import _sanitize_text
        return {
            "chunk_id": data["chunk_id"],
            "text": _sanitize_text(data["text"]),
            "heading": _sanitize_text(data.get("heading", "")),
            "order": data["order"],
            "doc_id": data["doc_id"],
            "title": _sanitize_text(data["title"] or "")
        }
    
    return [(sanitize_chunk_data(r), float(r["score"])) for r in rows or []]

def hybrid_search(driver, q_vec: List[float], q_text: str, top_k: int = 12,
                  rrf_k: int = 60) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion:
      score(doc) = sum(1 / (rrf_k + rank_i(doc))) over rankers i
    Always returns a list (possibly empty).
    """
    try:
        v = vector_search(driver, q_vec, top_k=top_k*3)
        k = keyword_search(driver, q_text, top_k=top_k*4)
    except Exception:
        return []

    def rankmap(pairs):
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        id2rank, id2rec = {}, {}
        for i, (rec, _) in enumerate(sorted_pairs, start=1):
            id2rank[rec["chunk_id"]] = i
            id2rec[rec["chunk_id"]]  = rec
        return id2rank, id2rec

    v_rank, v_rec = rankmap(v)
    k_rank, k_rec = rankmap(k)

    ids = set(v_rank) | set(k_rank)
    if not ids:
        return []

    rrf = {}
    for cid in ids:
        s = 0.0
        if cid in v_rank: s += 1.0 / (rrf_k + v_rank[cid])
        if cid in k_rank: s += 1.0 / (rrf_k + k_rank[cid])
        rrf[cid] = s

    ranked = sorted(ids, key=lambda cid: rrf[cid], reverse=True)[:top_k]
    out: List[Dict[str, Any]] = []
    for cid in ranked:
        out.append(v_rec.get(cid) or k_rec[cid])
    return out
