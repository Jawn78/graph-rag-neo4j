"""
Reranking module for improved retrieval quality.

Implements:
- Cross-encoder reranking using embedding similarity
- Maximal Marginal Relevance (MMR) for diversity
- Score normalization and combination
"""

import os
import logging
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Reranking configuration
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "1") == "1"
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "20"))  # Candidates to rerank

# MMR configuration
# Lambda controls relevance vs diversity tradeoff
# 1.0 = pure relevance, 0.0 = pure diversity
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.7"))
MMR_ENABLED = os.getenv("MMR_ENABLED", "1") == "1"


@dataclass
class RankedChunk:
    """A chunk with ranking scores."""
    chunk: Dict[str, Any]
    relevance_score: float
    diversity_score: float = 0.0
    final_score: float = 0.0

    def __post_init__(self):
        if self.final_score == 0.0:
            self.final_score = self.relevance_score


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def _prefetch_embeddings(chunks: List[Dict[str, Any]], driver=None) -> Dict[str, List[float]]:
    """
    Resolve embeddings for a list of chunks with at most ONE database query.

    Embeddings already present on the chunk dicts are used as-is; the rest are
    fetched in a single batched Cypher query (instead of one round-trip per
    chunk, which dominated rerank latency).

    Returns dict of chunk_id -> embedding (chunks without one are absent).
    """
    result: Dict[str, List[float]] = {}
    missing: List[str] = []

    for chunk in chunks:
        if chunk.get("embedding"):
            result[chunk["chunk_id"]] = chunk["embedding"]
        else:
            missing.append(chunk["chunk_id"])

    if not missing or driver is None:
        return result

    try:
        from neo4j import READ_ACCESS
        from ..config import NEO4J_DB
        with driver.session(database=NEO4J_DB, default_access_mode=READ_ACCESS) as s:
            rows = s.run("""
                MATCH (c:Chunk) WHERE c.chunk_id IN $ids
                RETURN c.chunk_id AS chunk_id, c.embedding AS embedding
            """, ids=missing).data()
        for r in rows:
            if r["embedding"]:
                result[r["chunk_id"]] = list(r["embedding"])
    except Exception as e:
        logger.debug(f"Batch embedding fetch failed for {len(missing)} chunks: {e}")

    return result


def rerank_by_embedding(query_embedding: List[float],
                        chunks: List[Dict[str, Any]],
                        driver=None,
                        top_k: int = RERANK_TOP_K) -> List[Dict[str, Any]]:
    """
    Rerank chunks by embedding similarity to query.

    This is a lightweight cross-encoder approximation using
    cosine similarity of embeddings.

    Args:
        query_embedding: The query embedding vector
        chunks: List of chunks to rerank
        driver: Optional Neo4j driver for embedding lookup
        top_k: Number of top results to return

    Returns:
        Reranked list of chunks
    """
    if not chunks or not query_embedding:
        return chunks

    embeddings = _prefetch_embeddings(chunks, driver)
    scored: List[Tuple[float, Dict[str, Any]]] = []

    for chunk in chunks:
        chunk_emb = embeddings.get(chunk["chunk_id"])
        if chunk_emb:
            score = _cosine_similarity(query_embedding, chunk_emb)
        else:
            # Fallback: use existing score or 0
            score = chunk.get("_score", 0.0)

        scored.append((score, chunk))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Return top_k
    return [chunk for _, chunk in scored[:top_k]]


def mmr_rerank(query_embedding: List[float],
               chunks: List[Dict[str, Any]],
               driver=None,
               lambda_param: float = MMR_LAMBDA,
               top_k: int = RERANK_TOP_K,
               relevance_overrides: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    """
    Maximal Marginal Relevance reranking.

    Balances relevance to query with diversity among selected chunks.

    MMR formula:
        MMR = argmax[λ * sim(d, q) - (1-λ) * max(sim(d, d_i))]
        where d_i are already selected documents

    Args:
        query_embedding: The query embedding vector
        chunks: List of chunks to rerank
        driver: Optional Neo4j driver for embedding lookup
        lambda_param: Balance between relevance (1.0) and diversity (0.0)
        top_k: Number of results to return
        relevance_overrides: Optional chunk_id -> relevance map (e.g. from a
            cross-encoder) used instead of embedding cosine for the relevance
            term; diversity still uses embeddings

    Returns:
        Reranked list of chunks with diversity
    """
    if not chunks or not query_embedding:
        return chunks

    if len(chunks) <= 1:
        return chunks

    # Get embeddings for all chunks (single batched query)
    embeddings = _prefetch_embeddings(chunks, driver)
    chunk_embeddings: List[Tuple[Dict[str, Any], Optional[List[float]]]] = [
        (chunk, embeddings.get(chunk["chunk_id"])) for chunk in chunks
    ]

    # Compute relevance scores
    relevance_scores: Dict[str, float] = {}
    for chunk, emb in chunk_embeddings:
        cid = chunk["chunk_id"]
        if relevance_overrides is not None and cid in relevance_overrides:
            relevance_scores[cid] = relevance_overrides[cid]
        elif emb:
            relevance_scores[cid] = _cosine_similarity(query_embedding, emb)
        else:
            relevance_scores[cid] = chunk.get("_score", 0.5)

    # MMR selection
    selected: List[Dict[str, Any]] = []
    selected_embeddings: List[List[float]] = []
    remaining = list(chunk_embeddings)

    while len(selected) < top_k and remaining:
        best_score = float('-inf')
        best_idx = 0

        for idx, (chunk, emb) in enumerate(remaining):
            cid = chunk["chunk_id"]
            relevance = relevance_scores.get(cid, 0.0)

            # Compute max similarity to already selected
            max_sim_to_selected = 0.0
            if selected_embeddings and emb:
                for sel_emb in selected_embeddings:
                    sim = _cosine_similarity(emb, sel_emb)
                    max_sim_to_selected = max(max_sim_to_selected, sim)

            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        # Add best chunk to selected
        best_chunk, best_emb = remaining.pop(best_idx)
        selected.append(best_chunk)
        if best_emb:
            selected_embeddings.append(best_emb)

    return selected


def rerank_chunks(query_embedding: List[float],
                  chunks: List[Dict[str, Any]],
                  driver=None,
                  use_mmr: bool = MMR_ENABLED,
                  lambda_param: float = MMR_LAMBDA,
                  top_k: int = RERANK_TOP_K,
                  query_text: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Main reranking function.

    Relevance comes from a cross-encoder when one is configured
    (CROSS_ENCODER_MODEL + the 'rerank' extra installed) and query_text is
    provided; otherwise from embedding cosine similarity. MMR diversity is
    applied on top in either case.

    Args:
        query_embedding: The query embedding vector
        chunks: List of chunks to rerank
        driver: Optional Neo4j driver for embedding lookup
        use_mmr: Whether to apply MMR diversity
        lambda_param: MMR lambda (relevance vs diversity)
        top_k: Number of results to return
        query_text: Raw query text, required for cross-encoder scoring

    Returns:
        Reranked list of chunks
    """
    if not RERANK_ENABLED:
        return chunks[:top_k]

    if not chunks:
        return []

    logger.debug(f"Reranking {len(chunks)} chunks (MMR={use_mmr}, lambda={lambda_param})")

    # Cross-encoder path: joint (query, chunk) scoring beats re-using the
    # bi-encoder embeddings the vector index already searched with.
    ce_relevance: Optional[Dict[str, float]] = None
    if query_text:
        from .cross_encoder import cross_encoder_scores
        scores = cross_encoder_scores(query_text, chunks)
        if scores is not None:
            # Min-max normalize so scores compose with MMR's [0,1] cosine terms
            lo, hi = min(scores), max(scores)
            spread = (hi - lo) or 1.0
            ce_relevance = {}
            for chunk, s in zip(chunks, scores):
                norm = (s - lo) / spread
                ce_relevance[chunk["chunk_id"]] = norm
                chunk["_ce_score"] = float(s)
            logger.debug("Using cross-encoder relevance for rerank")

    if use_mmr:
        result = mmr_rerank(query_embedding, chunks, driver, lambda_param, top_k,
                            relevance_overrides=ce_relevance)
    elif ce_relevance is not None:
        ordered = sorted(chunks, key=lambda c: ce_relevance[c["chunk_id"]], reverse=True)
        result = ordered[:top_k]
    else:
        result = rerank_by_embedding(query_embedding, chunks, driver, top_k)

    logger.debug(f"Reranking complete: {len(result)} chunks returned")
    return result


def compute_diversity_score(chunks: List[Dict[str, Any]], driver=None) -> float:
    """
    Compute average pairwise diversity among chunks.

    Returns a score from 0 (identical) to 1 (maximally diverse).
    Useful for evaluating retrieval quality.
    """
    if len(chunks) < 2:
        return 1.0

    by_id = _prefetch_embeddings(chunks, driver)
    embeddings = [by_id[c["chunk_id"]] for c in chunks if c["chunk_id"] in by_id]

    if len(embeddings) < 2:
        return 1.0

    # Compute average pairwise similarity
    total_sim = 0.0
    count = 0
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            total_sim += _cosine_similarity(embeddings[i], embeddings[j])
            count += 1

    avg_similarity = total_sim / count if count > 0 else 0.0

    # Diversity is inverse of similarity
    return 1.0 - avg_similarity
