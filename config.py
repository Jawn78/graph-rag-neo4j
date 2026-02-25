import os
import threading
from dotenv import load_dotenv
from openai import OpenAI
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

load_dotenv()

# ---- Neo4j ----
# For a single-node local DB, use bolt://
NEO4J_URI   = os.getenv("NEO4J_URI", "bolt://192.168.50.205:7687")
NEO4J_USER  = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS  = os.getenv("NEO4J_PASS", "H@rtf0rdR3x")
NEO4J_DB    = os.getenv("NEO4J_DB", "neo4j")           # must match your Browser DB name
NEO4J_ENCRYPTED = os.getenv("NEO4J_ENCRYPTED", "0") == "1"

# Thread-safe driver initialization
_driver_instance = None
_driver_lock = threading.Lock()


def get_driver():
    """
    Return a verified neo4j driver with connection pooling.

    Thread-safe: uses double-checked locking pattern to ensure
    only one driver instance is created even under concurrent access.
    Auto-falls back from neo4j:// to bolt:// if needed.
    """
    global _driver_instance

    # Fast path: driver already initialized
    if _driver_instance is not None:
        return _driver_instance

    # Slow path: acquire lock and check again (double-checked locking)
    with _driver_lock:
        # Another thread may have initialized while we waited
        if _driver_instance is not None:
            return _driver_instance

        try:
            driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USER, NEO4J_PASS),
                encrypted=NEO4J_ENCRYPTED,
                max_connection_lifetime=3600,  # 1 hour
                max_connection_pool_size=50,   # Increased pool size
                connection_acquisition_timeout=30,  # 30 seconds
                keep_alive=True
            )
            driver.verify_connectivity()
            _driver_instance = driver
        except ServiceUnavailable:
            if NEO4J_URI.startswith("neo4j://"):
                alt = "bolt://" + NEO4J_URI.split("://", 1)[1]
                driver = GraphDatabase.driver(
                    alt,
                    auth=(NEO4J_USER, NEO4J_PASS),
                    encrypted=NEO4J_ENCRYPTED,
                    max_connection_lifetime=3600,
                    max_connection_pool_size=50,
                    connection_acquisition_timeout=30,
                    keep_alive=True
                )
                driver.verify_connectivity()
                _driver_instance = driver
            else:
                raise

    return _driver_instance


def close_driver():
    """Close the global driver instance. Thread-safe."""
    global _driver_instance
    with _driver_lock:
        if _driver_instance:
            _driver_instance.close()
            _driver_instance = None

# ---- Files ----
DOCS_DIR = os.getenv("DOCS_DIR", os.path.join(os.path.dirname(__file__), "Rag_Docs"))

# ---- Servers / models ----
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "http://127.0.0.1:8080/v1")
CHAT_BASE_URL  = os.getenv("CHAT_BASE_URL",  "http://127.0.0.1:8081/v1")
EMBED_MODEL    = os.getenv("EMBED_MODEL",    "qwen3-embed-0.6b")
# CHAT_MODEL     = os.getenv("CHAT_MODEL",     "mistral-7b-instruct-v0.2")
CHAT_MODEL     = os.getenv("CHAT_MODEL",     "llama-2-7b-chat")
# ---- MCP (Microsoft Content Protocol) server ----
# Optional MCP server that can be used for chat/completions. Enable by setting MCP_BASE_URL
# and MCP_CHAT_MODEL (or via the --use-mcp flag at runtime).
MCP_BASE_URL   = os.getenv("MCP_BASE_URL", "https://learn.microsoft.com/api/mcp")
MCP_CHAT_MODEL = os.getenv("MCP_CHAT_MODEL", "mcp-default-model")
USE_MCP_BY_DEFAULT = os.getenv("USE_MCP_BY_DEFAULT", "0") == "1"
MCP_AGENT_URL  = os.getenv("MCP_AGENT_URL", "")  # URL of an agent gateway that speaks to the Learn MCP server

# ---- Index ----
CHUNK_INDEX_NAME = os.getenv("CHUNK_INDEX_NAME", "chunk_embedding_index")

# ---- Performance Configuration ----
PARSE_WORKERS = int(os.getenv("PARSE_WORKERS", "8"))
EMBED_WORKERS = int(os.getenv("EMBED_WORKERS", "4"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
ENABLE_PERFORMANCE_MONITORING = os.getenv("ENABLE_PERFORMANCE_MONITORING", "1") == "1"

# ---- Clients ----
embed_client = OpenAI(base_url=EMBED_BASE_URL, api_key=os.getenv("EMBED_API_KEY", "llamacpp"))
chat_client  = OpenAI(base_url=CHAT_BASE_URL,  api_key=os.getenv("CHAT_API_KEY",  "llamacpp"))
# MCP client: create but do not replace chat_client unless caller requests it
mcp_client   = OpenAI(base_url=MCP_BASE_URL,   api_key=os.getenv("MCP_API_KEY", ""))

def check_mcp_health(timeout: float = 5.0) -> tuple[bool, str]:
    """Quick health check for the MCP endpoint: try a minimal chat request or a GET to base URL.
    Returns (ok, message). This function must be conservative and not raise on connection errors.
    """
    try:
        # Try a lightweight info call if supported; otherwise attempt a GET to the base URL
        # Many OpenAI-compatible endpoints don't expose a simple /health; we attempt a simple request
        # that should return 404 if model not found (which still indicates reachable), so we treat
        # 2xx or 404 as "reachable" and anything else as failure.
        from openai import OpenAI
        # Use the client but do not assume any model exists; create a minimal request and catch errors
        try:
            import requests
            resp = requests.get(MCP_BASE_URL, timeout=timeout)
            # if we got any response code, consider reachable
            return True, f"HTTP GET to MCP_BASE_URL returned: {resp.status_code}"
        except Exception as e:
            # If .get isn't available or errors, attempt a minimal chat request to the configured model
            try:
                mcp_client.chat.completions.create(
                    model=MCP_CHAT_MODEL,
                    messages=[{"role":"system","content":"ping"}],
                    max_tokens=1,
                )
                return True, "MCP chat request succeeded"
            except Exception as e2:
                # Consider 404 as reachable but missing model
                msg = repr(e2)
                if "404" in msg or "NotFound" in msg or "Not Found" in msg:
                    return True, f"MCP reachable but returned 404/NotFound: {msg}"
                return False, f"MCP unreachable or errored: {msg}"
    except Exception as e:
        return False, f"MCP health check error: {repr(e)}"

def detect_embedding_dim() -> int:
    vec = embed_client.embeddings.create(
        model=EMBED_MODEL, input="dimension probe", encoding_format="float"
    ).data[0].embedding
    if not vec:
        raise RuntimeError("Embedding server returned empty embedding.")
    return len(vec)
