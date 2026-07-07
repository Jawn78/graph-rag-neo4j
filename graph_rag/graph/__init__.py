"""Neo4j graph operations: schema, upserts, search, cleanup."""

from typing import Dict, Optional, Union

from .schema import ensure_schema
from .query import vector_search, keyword_search, hybrid_search
from .upsert import upsert_docs, upsert_chunks

__all__ = [
    "ensure_schema",
    "vector_search",
    "keyword_search",
    "hybrid_search",
    "upsert_docs",
    "upsert_chunks",
    "check_neo4j_connection",
]


def check_neo4j_connection() -> Dict[str, Union[bool, Optional[str]]]:
    """Check whether the configured Neo4j server is reachable."""
    try:
        from ..config import get_driver

        driver = get_driver()
        driver.verify_connectivity()
        return {"connected": True, "error": None}
    except Exception as e:
        return {"connected": False, "error": str(e)}
