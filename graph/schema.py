import re
from typing import List, Optional
from neo4j import Driver

from ..config import CHUNK_INDEX_NAME, NEO4J_DB

def _validate_index_name(name: str) -> str:
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        raise ValueError(f"Invalid index name: {name}")
    return name

def _fulltext_props(s, name: str) -> Optional[List[str]]:
    """Return the properties array for a given index name, or None if not found."""
    row = s.run(
        """
        SHOW INDEXES
        YIELD name, type, entityType, labelsOrTypes, properties
        WHERE type = 'FULLTEXT' AND name = $name AND entityType = 'NODE' AND labelsOrTypes = ['Chunk']
        RETURN properties
        """,
        name=name,
    ).single()
    return None if row is None else row["properties"]

def ensure_schema(driver: Driver, embed_dim: int) -> None:
    """
    Creates/repairs:
      - Uniqueness constraints on Document.doc_id and Chunk.chunk_id
      - B-tree index on Document.title
      - FULLTEXT index 'chunk_text_fts' on Chunk(text, heading)
      - VECTOR index on Chunk(embedding) with COSINE and the detected dimension
    """
    with driver.session(database=NEO4J_DB) as s:
        # --- Constraints / b-tree ---
        s.run("CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE")
        s.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE")
        s.run("CREATE INDEX doc_title IF NOT EXISTS FOR (d:Document) ON (d.title)")

        # --- FULLTEXT: ensure it covers BOTH 'text' and 'heading' ---
        fts_name = "chunk_text_fts"
        try:
            props = _fulltext_props(s, fts_name)
            if props is None:
                # Create from scratch (DDL)
                s.run(
                    """
                    CREATE FULLTEXT INDEX chunk_text_fts IF NOT EXISTS
                    FOR (c:Chunk) ON EACH [c.text, c.heading]
                    """
                )
            else:
                want = {"text", "heading"}
                have = set(props or [])
                if not want.issubset(have):
                    # Drop & recreate to add 'heading'
                    s.run("DROP INDEX chunk_text_fts IF EXISTS")
                    s.run(
                        """
                        CREATE FULLTEXT INDEX chunk_text_fts
                        FOR (c:Chunk) ON EACH [c.text, c.heading]
                        """
                    )
        except Exception:
            # Fallback to legacy procedure path
            props = _fulltext_props(s, fts_name)
            if props is None:
                s.run(
                    "CALL db.index.fulltext.createNodeIndex($name, ['Chunk'], ['text','heading'])",
                    name=fts_name,
                )
            else:
                want = {"text", "heading"}
                have = set(props or [])
                if not want.issubset(have):
                    s.run("CALL db.index.fulltext.drop($name)", name=fts_name)
                    s.run(
                        "CALL db.index.fulltext.createNodeIndex($name, ['Chunk'], ['text','heading'])",
                        name=fts_name,
                    )

        # --- VECTOR index on Chunk(embedding) ---
        vec_exists = s.run(
            "SHOW INDEXES YIELD name WHERE name = $n RETURN count(*) AS c",
            n=CHUNK_INDEX_NAME,
        ).single()["c"]
        if vec_exists == 0:
            idx = _validate_index_name(CHUNK_INDEX_NAME)
            s.run(
                f"""
                CREATE VECTOR INDEX {idx} IF NOT EXISTS
                FOR (c:Chunk) ON (c.embedding)
                OPTIONS {{
                  indexConfig: {{
                    `vector.dimensions`: $dim,
                    `vector.similarity_function`: 'COSINE'
                  }}
                }}
                """,
                dim=embed_dim,
            )
