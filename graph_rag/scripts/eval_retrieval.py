#!/usr/bin/env python3
"""
Retrieval regression evaluation from positive user feedback.

Builds a golden set from thumbs-up / high-rating feedback entries (each pairs
a query with the chunk_ids that produced a good answer), re-runs retrieval
for every query, and reports hit-rate@k and MRR. Run this before and after
tuning retrieval (heading boost, MMR lambda, cross-encoder, chunking) to see
whether the change actually helped.

Usage:
    python -m graph_rag.scripts.eval_retrieval [--top-k 6] [--days 90]
    python -m graph_rag.scripts.eval_retrieval --golden golden.jsonl

A golden JSONL file (one {"query": ..., "chunk_ids": [...]} per line) can be
used instead of the feedback DB, e.g. for a hand-curated suite.
"""

import argparse
import json
import sqlite3
import sys
from datetime import timedelta
from typing import Dict, List, Tuple


def load_golden_from_feedback(db_path: str, days: int, min_value: float = 0.75) -> List[Tuple[str, List[str]]]:
    """Load (query, expected_chunk_ids) pairs from positive feedback."""
    from graph_rag.context_engine.types import utcnow

    cutoff = (utcnow() - timedelta(days=days)).isoformat()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT query, chunk_ids FROM feedback
            WHERE value >= ? AND chunk_ids IS NOT NULL AND chunk_ids != ''
              AND feedback_type IN ('thumbs_up', 'rating')
              AND timestamp > ?
            """,
            (min_value, cutoff),
        ).fetchall()
    finally:
        conn.close()

    golden: Dict[str, List[str]] = {}
    for r in rows:
        try:
            chunk_ids = json.loads(r["chunk_ids"])
        except (json.JSONDecodeError, TypeError):
            continue
        if r["query"] and chunk_ids:
            # Last positive feedback for a query wins
            golden[r["query"]] = list(chunk_ids)

    return list(golden.items())


def load_golden_from_file(path: str) -> List[Tuple[str, List[str]]]:
    """Load golden pairs from a JSONL file."""
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("query") and obj.get("chunk_ids"):
                pairs.append((obj["query"], list(obj["chunk_ids"])))
    return pairs


def evaluate(golden: List[Tuple[str, List[str]]], top_k: int) -> Dict[str, float]:
    """Re-run retrieval for each golden query and score against expectations."""
    from graph_rag.config import get_driver
    from graph_rag.qa.answer import _embed_query, _retrieve_chunks

    driver = get_driver()

    hits = 0
    reciprocal_ranks: List[float] = []
    evaluated = 0

    for query, expected_ids in golden:
        q_emb = _embed_query(query, trace_id="eval")
        if q_emb is None:
            print(f"  SKIP (embed failed): {query[:60]}")
            continue

        retrieved = _retrieve_chunks(driver, query, q_emb, top_k, trace_id="eval")
        retrieved_ids = [c["chunk_id"] for c in retrieved[:top_k]]
        expected = set(expected_ids)

        evaluated += 1
        rank = next((i + 1 for i, cid in enumerate(retrieved_ids) if cid in expected), None)
        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
            print(f"  HIT@{rank}: {query[:60]}")
        else:
            reciprocal_ranks.append(0.0)
            print(f"  MISS:  {query[:60]}")

    if not evaluated:
        return {"queries": 0, "hit_rate": 0.0, "mrr": 0.0}

    return {
        "queries": evaluated,
        "hit_rate": hits / evaluated,
        "mrr": sum(reciprocal_ranks) / evaluated,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval against golden query/chunk pairs")
    parser.add_argument("--top-k", type=int, default=6, help="Retrieval depth to evaluate at")
    parser.add_argument("--days", type=int, default=90, help="Feedback window for golden pairs")
    parser.add_argument("--golden", help="JSONL file of {query, chunk_ids} pairs (bypasses feedback DB)")
    parser.add_argument("--feedback-db", default=None, help="Path to feedback.db (default: FEEDBACK_DB_PATH)")
    args = parser.parse_args()

    if args.golden:
        golden = load_golden_from_file(args.golden)
        source = args.golden
    else:
        from graph_rag.context_engine.feedback import FEEDBACK_DB_PATH
        db_path = args.feedback_db or FEEDBACK_DB_PATH
        golden = load_golden_from_feedback(db_path, args.days)
        source = db_path

    if not golden:
        print(f"No golden pairs found in {source}.")
        print("Collect thumbs-up feedback (submit_feedback) or provide --golden file.")
        return 1

    print(f"Evaluating {len(golden)} golden queries (top-k={args.top_k})...\n")
    metrics = evaluate(golden, args.top_k)

    print("\n=== Results ===")
    print(f"Queries evaluated: {metrics['queries']}")
    print(f"Hit rate@{args.top_k}:  {metrics['hit_rate']:.1%}")
    print(f"MRR:               {metrics['mrr']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
