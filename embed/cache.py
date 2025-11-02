import sqlite3, json, os
from ..config import EMBED_MODEL
DB = os.getenv("EMBED_CACHE_DB", "embed_cache.sqlite")

def _ensure():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS cache(
        chunk_id TEXT NOT NULL,
        model    TEXT NOT NULL,
        dim      INTEGER NOT NULL,
        vec_json TEXT NOT NULL,
        PRIMARY KEY (chunk_id, model)
    )""")
    con.commit(); con.close()

def get(chunk_id: str):
    _ensure()
    con = sqlite3.connect(DB)
    row = con.execute("SELECT vec_json FROM cache WHERE chunk_id=? AND model=?",
                      (chunk_id, EMBED_MODEL)).fetchone()
    con.close()
    return json.loads(row[0]) if row else None

def put(chunk_id: str, vec: list[float]):
    _ensure()
    con = sqlite3.connect(DB)
    con.execute("INSERT OR REPLACE INTO cache(chunk_id, model, dim, vec_json) VALUES(?,?,?,?)",
                (chunk_id, EMBED_MODEL, len(vec), json.dumps(vec, separators=(',',':'))))
    con.commit(); con.close()
