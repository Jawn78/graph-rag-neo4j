"""
Optional cross-encoder reranking.

The embedding "rerank" in reranker.py scores query and chunk independently
with the same bi-encoder the vector index already used, so it adds little new
signal. A cross-encoder scores the (query, chunk) pair jointly and is the
standard relevance upgrade for RAG.

Enable by installing the extra and setting the model:

    pip install "graph-rag-neo4j[rerank]"
    export CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

When the package or model is unavailable this module reports itself disabled
and reranker.py falls back to embedding-based reranking unchanged.
"""

import os
import logging
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Empty = disabled. Suggested: cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, CPU-ok)
# or BAAI/bge-reranker-base (stronger).
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "")

# Truncate chunk text fed to the cross-encoder (they have short max lengths)
CE_MAX_CHUNK_CHARS = int(os.getenv("CE_MAX_CHUNK_CHARS", "1000"))

_model = None
_model_failed = False
_model_lock = threading.Lock()


def cross_encoder_available() -> bool:
    """True if a cross-encoder model is configured and loadable."""
    return bool(CROSS_ENCODER_MODEL) and _get_model() is not None


def _get_model():
    """Lazily load the sentence-transformers CrossEncoder. Cached; one attempt."""
    global _model, _model_failed
    if _model is not None or _model_failed or not CROSS_ENCODER_MODEL:
        return _model

    with _model_lock:
        if _model is not None or _model_failed:
            return _model
        try:
            from sentence_transformers import CrossEncoder
            _model = CrossEncoder(CROSS_ENCODER_MODEL)
            logger.info(f"Loaded cross-encoder: {CROSS_ENCODER_MODEL}")
        except ImportError:
            _model_failed = True
            logger.warning(
                "CROSS_ENCODER_MODEL is set but sentence-transformers is not "
                "installed; falling back to embedding rerank. "
                "Install with: pip install 'graph-rag-neo4j[rerank]'"
            )
        except Exception as e:
            _model_failed = True
            logger.warning(f"Failed to load cross-encoder '{CROSS_ENCODER_MODEL}': {e}")
    return _model


def cross_encoder_scores(query_text: str,
                         chunks: List[Dict[str, Any]]) -> Optional[List[float]]:
    """
    Score (query, chunk) pairs jointly. Returns one score per chunk, or None
    if the cross-encoder is unavailable or scoring fails.
    """
    if not query_text or not chunks:
        return None

    model = _get_model()
    if model is None:
        return None

    try:
        pairs = [
            (query_text, (c.get("text") or "")[:CE_MAX_CHUNK_CHARS])
            for c in chunks
        ]
        scores = model.predict(pairs)
        return [float(s) for s in scores]
    except Exception as e:
        logger.warning(f"Cross-encoder scoring failed: {e}")
        return None
