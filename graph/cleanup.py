from neo4j import Driver
from ..config import NEO4J_DB

def delete_all(driver: Driver) -> None:
    with driver.session(database=NEO4J_DB) as s:
        s.run("MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk) DETACH DELETE c")
        s.run("MATCH (d:Document) DETACH DELETE d")
        s.run("MATCH (e:Entity) DETACH DELETE e")

def delete_by_source(driver: Driver, source: str) -> int:
    with driver.session(database=NEO4J_DB) as s:
        rec = s.run("""
        MATCH (d:Document {source:$source})
        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
        DETACH DELETE d, c
        RETURN count(*) AS n
        """, source=source).single()
        return rec["n"] if rec else 0

def delete_by_folder_prefix(driver: Driver, prefix: str) -> int:
    with driver.session(database=NEO4J_DB) as s:
        rec = s.run("""
        WITH replace(toLower($prefix), "\\\\", "/") AS p
        MATCH (d:Document)
        WHERE d.path IS NOT NULL AND replace(toLower(d.path), "\\\\", "/") CONTAINS p
        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
        DETACH DELETE d, c
        RETURN count(*) AS n
        """, prefix=prefix).single()
        return rec["n"] if rec else 0

def delete_orphan_entities(driver: Driver) -> int:
    with driver.session(database=NEO4J_DB) as s:
        rec = s.run("""
        MATCH (e:Entity)
        WHERE NOT (e)--()
        DELETE e
        RETURN count(e) AS n
        """).single()
        return rec["n"] if rec else 0
