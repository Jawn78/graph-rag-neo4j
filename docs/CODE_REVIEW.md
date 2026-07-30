# Code Review: graph-rag-neo4j

Full-repository review covering correctness, architecture, security, and maintainability.
Complements `docs/PERFORMANCE_CRITIQUE.md`, which focused on performance.

---

## 1. Critical — fix these first

### 1.1 Hardcoded database credentials in `config.py`

`config.py:12-14` ships a real-looking password and a private LAN IP as defaults:

```python
NEO4J_URI  = os.getenv("NEO4J_URI", "bolt://192.168.50.205:7687")
NEO4J_PASS = os.getenv("NEO4J_PASS", "H@rtf0rdR3x")
```

This credential is now in the public git history permanently. Actions:

1. **Rotate the Neo4j password immediately** — removing it from the code does not remove it from history.
2. Make the defaults safe: default `NEO4J_URI` to `bolt://localhost:7687` and make `NEO4J_PASS` **required** (fail fast with a clear error if unset) rather than defaulting to a secret.
3. Add a committed `.env.example` documenting every env var with placeholder values.

### 1.2 A committed virtualenv makes up 98% of the repository

`git ls-files` shows **5,821 of 5,902 tracked files are `.venv-server/`** (a Windows virtualenv, including `.exe` files and full site-packages). `.gitignore` has `.venv` but that pattern does not match `.venv-server`.

```bash
git rm -r --cached .venv-server __pycache__ graph/__pycache__ ingest/__pycache__ qa/__pycache__
```

and change the `.gitignore` entry to `.venv*/`. Also note `__pycache__/main.cpython-311.pyc` is tracked but **`main.py` itself was never committed** — the real entry point for the pipeline exists only as orphaned bytecode (see 2.1).

### 1.3 Broken submodule gitlinks

`llama.cpp` and `llama-cpp-python` are committed as gitlink entries but there is no `.gitmodules`, so `git submodule status` fails and both directories are empty on a fresh clone. Either add a proper `.gitmodules`, or (better, since these are only needed for local GPU builds) remove the gitlinks and document the clone step in `BUILD_WITH_CUDA.md`.

---

## 2. Architecture — the repo is two disconnected projects

### 2.1 The CLI runs stubs; the real pipeline has no entry point

There are two parallel implementations that never talk to each other:

- **`graph_rag/`** — the installable package (`pyproject.toml` packages only this) with a polished Click CLI (`graph_rag/__main__.py`). But its `ingest` and `ask` commands call `DocumentIngestor` and `QuestionAnswerer`, which are **`# TODO` placeholders returning fake results** (`graph_rag/ingest/__init__.py:20`, `graph_rag/qa/__init__.py:18`). A user running `python -m graph_rag ask -q "..."` gets `"This is a placeholder answer."` presented as success.
- **Top-level packages `qa/`, `graph/`, `ingest/`, `embed/`, `utils/`, `context_engine/`, `config.py`** — the real, substantial implementation (hybrid search, RRF, ingestion, context engine). These all use relative imports (`from ..config import ...`), which only work when the repo root is itself a package — but there is no root `__init__.py` and no committed `main.py`. **As committed, the real pipeline cannot be imported or run at all.**

**Recommendation (highest-value refactor):** move `qa/`, `graph/`, `ingest/`, `embed/`, `utils/`, `context_engine/`, and `config.py` *into* the `graph_rag/` package, delete the stub `DocumentIngestor`/`QuestionAnswerer`, and wire the CLI to the real `ingest_folder()`/`ask()` functions. One package, one entry point, `pip install -e .` works, and the relative imports become valid.

### 2.2 Three conflicting sources of Neo4j configuration

| File | URI default | Password default |
|---|---|---|
| `config.py` | `bolt://192.168.50.205:7687` | (leaked secret) |
| `setup/config.py` | — (no driver) | — |
| `graph_rag/setup/config.py` | `bolt://localhost:7687` | `password` |

`graph_rag/setup/config.py:get_driver()` also creates a **new unpooled driver per call** (and `clean_corrupted.py` uses it as a context manager), while `config.py:get_driver()` maintains a proper singleton with pooling and fallback. Keep exactly one config module and one `get_driver()`.

### 2.3 Duplicated `setup/` packages

`setup/` and `graph_rag/setup/` are near-copies of each other (`cli.py`, `config.py`, `environment.py`, `server.py` all differ only slightly — e.g., one `parent` in a path). Divergent copies guarantee they rot. Delete the top-level `setup/` and keep `graph_rag/setup/`.

### 2.4 Redundant embedding clients

`embed/client.py` creates its own `OpenAI` client with a hardcoded `api_key="llamacpp"`, duplicating `config.embed_client` and `config.detect_embedding_dim()`. Delete the module.

---

## 3. Bugs (verified in code)

### 3.1 Heading bleeds across documents during ingestion — `ingest/files.py:284`

```python
def _build_chunks(docs):
    out = []
    current_heading = ""        # <-- initialized OUTSIDE the doc loop
    for d in docs:
        ...
```

`current_heading` is never reset per document, so the last heading of document A is attached to the leading chunks of document B whenever B's first chunks have no detected heading. Since headings feed the FTS index and the answer-time heading boost, this silently mis-ranks retrieval. Move `current_heading = ""` inside the `for d in docs:` loop.

### 3.2 `_embed_in_place` can fire-and-forget — `ingest/files.py:428-437`

```python
loop = asyncio.get_event_loop()
if loop.is_running():
    asyncio.create_task(_embed_chunks_async(chunks))   # returns immediately!
```

If this is ever called from an async context, the task is created and **not awaited**, so `upsert_chunks()` runs with `embedding=None` for every chunk and the vectors never reach Neo4j. Also, `asyncio.get_event_loop()` is deprecated. Since callers are synchronous, replace the whole dance with `asyncio.run(_embed_chunks_async(chunks))` and let it raise if called from an async context.

### 3.3 Wrong constant for session access mode — `qa/answer.py:244`

```python
with driver.session(database=NEO4J_DB, default_access_mode=RoutingControl.READ) as s:
```

`RoutingControl.READ` (value `"r"`) is the enum for `driver.execute_query()`; session config expects `neo4j.READ_ACCESS` (`"READ"`). Because every query here is wrapped in a broad `except Exception`, a driver that rejects the value will just log a warning and return `None` — the FTS anchor feature silently disappears. Use `default_access_mode=neo4j.READ_ACCESS` (and apply READ mode consistently in `graph/query.py`, which currently uses default WRITE sessions for pure reads).

### 3.4 `_score` is consumed everywhere but produced nowhere

`hybrid_search()` computes RRF scores and then **drops them** — the returned chunk dicts contain no score. Downstream, the Phase 2–4 machinery assumes `chunk["_score"]` exists:

- `qa/answer.py:966` — personalization re-sort: `c.get("_score", 0) * boost` → every key is `0`, the sort is a no-op.
- `context_engine/profiles.py:403` — boost only applied `if "_score" in chunk` → never.
- `context_engine/reranker.py:116,170` — fallback relevance is always the default.
- `context_engine/feedback.py:484` — `avg_chunk_score` analytics logs 0.0 forever.

**Fix:** have `vector_search`/`keyword_search`/`hybrid_search` attach `_score` (RRF score) to each returned chunk. Right now large parts of "Phases 2–4" are silently dead code in production paths.

### 3.5 Filters operate on metadata the retrieval layer never returns

`apply_filters()` / `apply_recency_boost()` (`context_engine/filters.py`) read `source`, `ext`, and date fields from chunk metadata — but the Cypher in `graph/query.py` returns only `chunk_id, text, heading, order, doc_id, title`. `metadata_json` is stored on nodes at ingest but never queried back. So user-preference filtering and recency boosting in `ask_full_context()` are also no-ops. Return (and parse) `metadata_json`, or join to the `Document` node's `source`/`path` properties in the retrieval queries.

### 3.6 Re-ingesting an edited document duplicates it forever

`doc_id = sha1(title + ":" + text[:2000])` (`ingest/files.py:263`). Any edit within the first 2000 chars produces a *new* doc_id; the old Document node and all of its chunks stay in the graph and keep matching queries with stale content. Meanwhile chunk IDs depend on chunk index and text prefix, so shifted chunk boundaries also strand old chunks. Use a stable doc identity (e.g., normalized path) plus a separate `content_hash` property, and on upsert delete chunks of that doc that aren't in the new chunk set (`graph/cleanup.py` has building blocks for this).

### 3.7 Packaging metadata is wrong

- `pyproject.toml` declares `requires-python = ">=3.8"` but the code uses `tuple[Path, ...]` / `list[float]` builtin generics evaluated at runtime (3.9+) and `X | None` unions elsewhere — install on 3.8 and it crashes at import.
- `name = "graph_rag2"` looks like a leftover; dependencies list only `click` and `rich`, omitting `neo4j`, `openai`, `pydantic`, `python-dotenv`, `requests`, `psutil`, etc. Consolidate `requirements*.txt` into `[project.dependencies]` / extras.

### 3.8 Misconceptions baked into the MCP integration

`config.py:92` calls MCP the "Microsoft Content Protocol" — it is the *Model Context Protocol*, and it is **not** an OpenAI-compatible REST API. `mcp_client = OpenAI(base_url="https://learn.microsoft.com/api/mcp")` can never work: the fallback tier `_call_mcp_client()` will fail on every request, adding latency to each `use_mcp=True` question before falling through. Either implement a real MCP client (the official `mcp` Python SDK) behind the existing gateway abstraction, or delete the dead tier and keep only `MCP_AGENT_URL` → chat fallback.

---

## 4. Refactoring recommendations

### 4.1 Collapse the three `ask*` functions (~600 duplicated lines)

`ask()`, `ask_with_context()`, and `ask_full_context()` in `qa/answer.py` re-implement the same pipeline: embed → FTS anchor → hybrid search → merge/dedupe → hydrate headings → heading boost → pack context → LLM fallback chain → sanitize. They have already drifted (only `ask()` lacks intent prompts; only `ask_full_context()` fetches extra candidates). Extract:

```python
def _retrieve(driver, query, q_emb, top_k, include_neighbors) -> list[Chunk]: ...
def _generate(messages, use_mcp, trace_id) -> tuple[str, str | None]: ...
```

and make `ask()` a thin call of the full pipeline with features disabled. One code path, one place to fix bugs like 3.4.

### 4.2 Kill the N+1 embedding fetch in the reranker

`_get_chunk_embedding()` (`context_engine/reranker.py:56`) opens a session and runs one Cypher query **per chunk**, and `mmr_rerank` + `compute_diversity_score` each do this over all candidates — with `rerank_multiplier = 3`, that's ~40 round-trips per question. Batch it:

```cypher
MATCH (c:Chunk) WHERE c.chunk_id IN $ids RETURN c.chunk_id, c.embedding
```

Better: return `node.embedding` directly from `vector_search` (the node is already in hand). Also use `numpy` for the cosine/MMR math instead of pure-Python loops over 1k-dim vectors — it's already an indirect dependency.

### 4.3 Reduce global singleton state

`config.py` instantiates three `OpenAI` clients at import time, and `_engine`, `_session_manager`, `ProfileManager`, `get_feedback_collector()` are module-level singletons. This makes unit testing impossible without monkeypatching and causes import-time side effects (e.g., `ProcessPoolExecutor` workers re-importing `config` spawn their own clients). Introduce a small `AppContext`/factory that owns clients + driver and is passed explicitly (the code already passes `driver` around — extend that pattern).

### 4.4 Simplify Cypher upserts

In `graph/upsert.py` both `ON CREATE SET` and `ON MATCH SET` set identical properties — replace with a single `SET` after `MERGE`. Consider `apoc`-free chunked transactions using the existing `BATCH_SIZE` config for very large ingests (embeddings make row payloads big).

### 4.5 Miscellaneous

- `LLMResponse.choices` / `_Choice` / `_Message` (`qa/answer.py:105-127`): `@dataclass` classes with hand-written `__init__` — the compatibility shim is unused; delete it or the dataclass decorators.
- `check_mcp_health()` (`config.py:116`) treats *any* HTTP response (including 500) as healthy and imports `requests`/`OpenAI` inside the function; simplify or remove.
- `_chunk_text()` never splits a single sentence longer than `target_chars`, so one run-on "sentence" (common in extracted PDFs) becomes a 12,000-char chunk. Add a hard character-window fallback.
- FTS query strings (`_fts_query_from_question`) are passed to Lucene unescaped; tokens like `c++` or stray `/` throw Lucene parse errors that are swallowed as "no results". Escape Lucene special characters.
- `graph_rag/scripts/test LLAMA CPP GPU Support.py` — rename (spaces make it unimportable and awkward in shells).
- `utils/performance.py` (`PerformanceMonitor`) and `utils/health_check.py` appear unused by the pipeline — wire them in or delete.
- `datetime.now()` in sessions/feedback is naive local time; prefer `datetime.now(timezone.utc)`.

---

## 5. Enhancements

1. **Tests — there are none.** Highest-leverage additions, in order: pure-function unit tests (`_chunk_text`, `_build_chunks` heading inheritance — would have caught 3.1; RRF fusion; `_apply_heading_boost`; `sanitize_text`/`is_corrupted`; MMR), then an integration test against Neo4j in Docker (testcontainers) covering ingest → search → answer with a stubbed LLM.
2. **CI** — a minimal GitHub Actions workflow: `ruff check`, `ruff format --check`, `mypy` (the codebase is already well-annotated), `pytest`. This also prevents regressions like committed venvs.
3. **Secret scanning** — enable GitHub secret scanning / pre-commit `detect-secrets`, given 1.1.
4. **Retrieval quality evaluation** — the feedback tables in `context_engine/feedback.py` are a good start; add a small golden-question eval script that reports hit-rate/MRR so retrieval changes (like fixing 3.4/3.5) can be measured.
5. **Config surface** — 30+ env vars are scattered across modules with duplicated defaults (`PARSE_WORKERS`, `EMBED_BATCH_SIZE` defined in both `config.py` and `ingest/files.py` — the `config.py` copies are unused). Centralize into one typed settings object (pydantic-settings is already in the dependency tree).
6. **Docs truth-check** — `README.md`/`SETUP.md` describe the `python -m graph_rag` CLI, which currently returns placeholder answers (2.1). Update once the packages are merged.

---

## 6. Suggested order of work

| # | Action | Effort | Impact |
|---|--------|--------|--------|
| 1 | Rotate Neo4j password; remove secret + private IP defaults; add `.env.example` | S | Critical |
| 2 | Untrack `.venv-server`, `__pycache__`; fix `.gitignore`; remove broken gitlinks | S | Critical |
| 3 | Merge top-level packages into `graph_rag/`, delete stubs, wire CLI to real pipeline | M | Critical |
| 4 | Fix heading-bleed bug (3.1) and access-mode bug (3.3) | S | High |
| 5 | Attach `_score` + metadata to retrieval results (3.4, 3.5) | S | High — revives Phases 2–4 |
| 6 | Extract shared retrieval pipeline from the three `ask*` functions | M | High |
| 7 | Batch reranker embedding fetch (4.2) | S | High (latency) |
| 8 | Stable doc identity + stale-chunk cleanup on re-ingest (3.6) | M | Medium |
| 9 | Tests + CI + ruff/mypy | M | High (long-term) |
| 10 | Fix packaging metadata & dependency declarations (3.7) | S | Medium |
