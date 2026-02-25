# Graph RAG with Neo4j - Technical Documentation

> A GPU-accelerated Retrieval-Augmented Generation system combining Neo4j graph database with local LLM inference for private, on-premise knowledge management.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Data Model](#data-model)
5. [Search & Retrieval](#search--retrieval)
6. [Configuration](#configuration)
7. [API Reference](#api-reference)
8. [Deployment](#deployment)

---

## Overview

### Purpose

Graph RAG is a knowledge management and question-answering system designed for organizations requiring:

- **Privacy**: All processing happens locally with no external API calls
- **Semantic Search**: Vector embeddings enable meaning-based retrieval
- **Structured Knowledge**: Neo4j graph database preserves document relationships
- **GPU Acceleration**: CUDA-enabled inference for production-grade performance

### Key Features

| Feature | Description |
|---------|-------------|
| Hybrid Search | Combines vector similarity with full-text keyword matching |
| Reciprocal Rank Fusion | Merges multiple ranking signals for better relevance |
| Heading-Based Boosting | Prioritizes chunks whose headings match query terms |
| Context Packing | Fits maximum relevant context within token budgets |
| Embedding Cache | SQLite-based caching eliminates redundant computation |
| Parallel Processing | Multi-threaded file parsing and embedding generation |
| Graceful Degradation | Fallback chain for MCP/chat client failures |

### Technology Stack

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                     │
│  Python 3.8+ │ Click CLI │ Rich Terminal UI             │
├─────────────────────────────────────────────────────────┤
│                     Inference Layer                      │
│  llama-cpp-python │ Qwen3-Embed │ Llama-2-7b-chat       │
│  CUDA 12.1+ │ GPU Acceleration                          │
├─────────────────────────────────────────────────────────┤
│                     Storage Layer                        │
│  Neo4j 5.20+ │ Vector Index │ Full-Text Index           │
│  SQLite Embedding Cache                                 │
├─────────────────────────────────────────────────────────┤
│                   Document Processing                    │
│  pypdf │ docx2txt │ chardet │ ThreadPoolExecutor        │
└─────────────────────────────────────────────────────────┘
```

---

## Architecture

### System Components

```
                                 ┌──────────────────┐
                                 │   User / CLI     │
                                 └────────┬─────────┘
                                          │
                           ┌──────────────┴──────────────┐
                           │                             │
                    ┌──────▼──────┐              ┌───────▼───────┐
                    │   Ingest    │              │   Question    │
                    │   Pipeline  │              │   Answering   │
                    └──────┬──────┘              └───────┬───────┘
                           │                             │
         ┌─────────────────┼─────────────────┐           │
         │                 │                 │           │
   ┌─────▼─────┐    ┌──────▼──────┐   ┌──────▼──────┐    │
   │   File    │    │   Chunk     │   │   Embed     │    │
   │  Parsers  │    │  Builder    │   │   Service   │◄───┤
   └───────────┘    └─────────────┘   └──────┬──────┘    │
                                             │           │
                                      ┌──────▼──────┐    │
                                      │   Embed     │    │
                                      │   Cache     │    │
                                      └─────────────┘    │
                                                         │
                    ┌────────────────────────────────────┤
                    │                                    │
             ┌──────▼──────┐                     ┌───────▼───────┐
             │   Neo4j     │                     │    Hybrid     │
             │  Database   │◄────────────────────│    Search     │
             └─────────────┘                     └───────┬───────┘
                                                         │
                                                 ┌───────▼───────┐
                                                 │    Context    │
                                                 │    Packer     │
                                                 └───────┬───────┘
                                                         │
                                                 ┌───────▼───────┐
                                                 │  Chat Server  │
                                                 │  (Llama-2)    │
                                                 └───────────────┘
```

### Data Flow

#### Ingestion Pipeline

```
1. Collect Files    ─→  Glob patterns: *.pdf, *.docx, *.txt, *.md, etc.
2. Parse Parallel   ─→  ThreadPoolExecutor (8 workers default)
3. Extract Text     ─→  pypdf, docx2txt, chardet fallback
4. Sanitize         ─→  Remove control chars, normalize Unicode
5. Chunk Text       ─→  1800 chars target, 200 char overlap
6. Extract Headings ─→  Regex patterns for sections/titles
7. Generate IDs     ─→  SHA1 hash of content
8. Cache Check      ─→  SQLite lookup by chunk_id
9. Embed Parallel   ─→  ThreadPoolExecutor (4 workers default)
10. Batch Embed     ─→  32 chunks per API call
11. Cache Store     ─→  Persist embeddings for reuse
12. Upsert Docs     ─→  MERGE into Neo4j (100 per batch)
13. Upsert Chunks   ─→  MERGE with vectors and relationships
```

#### Query Pipeline

```
1. Embed Question   ─→  Convert to vector via embedding service
2. Prepare FTS      ─→  Extract tokens + bigram phrases
3. Find Anchor      ─→  Best full-text match
4. Neighbor Window  ─→  ±2 chunks around anchor
5. Hybrid Search    ─→  Vector + keyword with RRF fusion
6. Deduplicate      ─→  Merge by chunk_id
7. Hydrate Headings ─→  Fill missing from database
8. Apply Boost      ─→  Re-rank by heading relevance
9. Pack Context     ─→  Fit within 2800 token budget
10. Generate Answer ─→  LLM chat completion with citations
```

---

## Core Components

### 1. Document Ingestion (`ingest/files.py`)

**Purpose**: Parse, chunk, and embed documents from various formats.

#### Supported Formats

| Extension | Library | Notes |
|-----------|---------|-------|
| `.pdf` | pypdf / PyPDF2 | Page-by-page extraction |
| `.docx` | docx2txt | Microsoft Word documents |
| `.txt`, `.md` | chardet | Encoding auto-detection |
| `.html`, `.htm` | chardet | Plain text extraction |
| `.csv`, `.rtf` | chardet | Basic text extraction |

#### Chunking Strategy

```python
CHUNK_TARGET_CHARS = 1800  # Target chunk size
CHUNK_OVERLAP = 200        # Overlap between chunks
EMBED_MAX_CHARS = 12000    # Maximum text for embedding
```

- Splits on sentence boundaries (`(?<=[\.\!\?])\s+`)
- Maintains overlap for context continuity
- Preserves heading context across chunks

#### Heading Detection

Patterns recognized as headings:
- Numbered sections: `1.2.3 Section Title`
- Markdown headers: `## Section Title`
- ALL CAPS lines: `SECTION TITLE`

#### Text Sanitization

The `_sanitize_for_embedding()` function handles:
- Control character removal
- Zero-width character stripping
- Unicode normalization
- Excessive repetition cleanup
- UTF-8 encoding enforcement

#### Key Functions

| Function | Description |
|----------|-------------|
| `ingest_folder(driver, folder)` | Main entry point; returns (docs, chunks) count |
| `_parse_files_parallel(paths, workers)` | Multi-threaded file parsing |
| `_build_chunks(docs)` | Create chunks with headings |
| `_embed_in_place(chunks)` | Parallel embedding with batching |

### 2. Neo4j Graph Operations (`graph/`)

#### Schema (`graph/schema.py`)

**Nodes:**
- `Document`: Represents source documents
- `Chunk`: Text segments with embeddings

**Relationships:**
- `(Document)-[:HAS_CHUNK]->(Chunk)`

**Indices:**
- `chunk_embedding_index`: Vector index (COSINE similarity)
- `chunk_text_fts`: Full-text index on `text` and `heading`

**Constraints:**
- Unique `Document.doc_id`
- Unique `Chunk.chunk_id`

#### Upsert Operations (`graph/upsert.py`)

```python
BATCH_SIZE = 100  # Documents/chunks per batch
```

- Uses `UNWIND` for efficient batch inserts
- `MERGE` operations for idempotency
- Sanitizes text before storage
- Stores metadata as JSON strings

#### Query Operations (`graph/query.py`)

| Function | Description |
|----------|-------------|
| `vector_search(driver, q_vec, top_k)` | Semantic similarity search |
| `keyword_search(driver, q_text, top_k)` | Full-text keyword search |
| `hybrid_search(driver, q_vec, q_text, top_k)` | Combined RRF search |

### 3. Embedding Service (`embed/`)

#### Client (`embed/client.py`)

OpenAI-compatible client wrapper for the local embedding server.

```python
EMBED_BASE_URL = "http://127.0.0.1:8080/v1"
EMBED_MODEL = "qwen3-embed-0.6b"
```

#### Cache (`embed/cache.py`)

SQLite-based caching indexed by `(chunk_id, model)`.

- Prevents redundant embedding computations
- Survives restarts for incremental ingestion
- Thread-safe operations

### 4. Question Answering (`qa/answer.py`)

#### System Prompt

```
You are a precise assistant. Use ONLY the provided context to answer.
- Be concise and factual.
- If the answer is not in the context, say you don't know.
- Cite supporting chunk(s) inline like [doc:title#chunk_order].
```

#### Context Packing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_CTX_TOKENS` | 2800 | Maximum context tokens |
| `MAX_OUTPUT_TOKENS` | 192 | Maximum response tokens |
| `PER_CHUNK_CHAR_CAP` | 900 | Character limit per chunk |
| `FOCUS_WINDOW` | 2 | Neighbors ± around anchor |
| `PER_DOC_MAX` | 3 | Maximum chunks per document |
| `DOC_ROUND_ROBIN` | true | Interleave chunks from different docs |

#### Heading Boost

Chunks are re-ranked based on heading relevance:

```python
HEADING_UNIT_BOOST = 0.12   # Boost per matching token
HEADING_BOOST_CAP = 0.45    # Maximum heading boost
HEADING_PHRASE_BONUS = 0.30 # Bonus for phrase matches
```

#### MCP Integration

The system supports Microsoft Content Protocol (MCP) with a fallback chain:

1. Agent Gateway (`MCP_AGENT_URL`) - preferred
2. MCP Client (`mcp_client`) - direct API
3. Default Chat Client (`chat_client`) - fallback

### 5. Server Management (`graph_rag/setup/server.py`)

Manages llama-cpp-python servers for embedding and chat.

| Server | Port | Model |
|--------|------|-------|
| Embedding | 8080 | qwen3-embed-0.6b |
| Chat | 8081 | llama-2-7b-chat |

Features:
- GPU layer allocation (`-1` for full GPU)
- Health checks with retry logic
- Context manager support
- Subprocess lifecycle management

---

## Data Model

### Neo4j Schema

```cypher
// Document Node
(:Document {
    doc_id: STRING,      // SHA1 hash (unique)
    title: STRING,       // Filename without extension
    metadata: STRING     // JSON: {source, path, filename, ext}
})

// Chunk Node
(:Chunk {
    chunk_id: STRING,    // SHA1 hash (unique)
    doc_id: STRING,      // Parent document ID
    order: INTEGER,      // Position in document
    text: STRING,        // Chunk content
    heading: STRING,     // Extracted/inherited heading
    embedding: FLOAT[]   // Vector embedding
})

// Relationship
(:Document)-[:HAS_CHUNK]->(:Chunk)

// Indices
CREATE VECTOR INDEX chunk_embedding_index FOR (c:Chunk) ON c.embedding
    OPTIONS {indexConfig: {`vector.dimensions`: 1024, `vector.similarity_function`: 'cosine'}}

CREATE FULLTEXT INDEX chunk_text_fts FOR (c:Chunk) ON EACH [c.text, c.heading]
```

### Embedding Dimensions

Auto-detected on startup via probe request. Common values:
- Qwen3-Embed-0.6b: 1024 dimensions

---

## Search & Retrieval

### Reciprocal Rank Fusion (RRF)

Combines vector and keyword search results:

```
score(doc) = Σ 1/(k + rank_i(doc)) for each ranker i
```

Where `k = 60` (default) dampens rank differences.

### Search Pipeline

```python
# 1. Vector search (3x top_k)
v = vector_search(driver, q_vec, top_k=top_k*3)

# 2. Keyword search (4x top_k)
k = keyword_search(driver, q_text, top_k=top_k*4)

# 3. Build rank maps
v_rank, v_rec = rankmap(v)
k_rank, k_rec = rankmap(k)

# 4. Compute RRF scores
for cid in ids:
    score = 0.0
    if cid in v_rank: score += 1.0 / (rrf_k + v_rank[cid])
    if cid in k_rank: score += 1.0 / (rrf_k + k_rank[cid])
    rrf[cid] = score

# 5. Sort and return top_k
ranked = sorted(ids, key=lambda cid: rrf[cid], reverse=True)[:top_k]
```

---

## Configuration

### Environment Variables

#### Neo4j Connection

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://192.168.50.205:7687` | Database URI |
| `NEO4J_USER` | `neo4j` | Username |
| `NEO4J_PASS` | (configured) | Password |
| `NEO4J_DB` | `neo4j` | Database name |
| `NEO4J_ENCRYPTED` | `0` | Enable TLS (`1` = yes) |

#### Model Services

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_BASE_URL` | `http://127.0.0.1:8080/v1` | Embedding server |
| `CHAT_BASE_URL` | `http://127.0.0.1:8081/v1` | Chat server |
| `EMBED_MODEL` | `qwen3-embed-0.6b` | Embedding model name |
| `CHAT_MODEL` | `llama-2-7b-chat` | Chat model name |

#### Performance Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `PARSE_WORKERS` | `8` | File parsing threads |
| `EMBED_WORKERS` | `4` | Embedding threads |
| `EMBED_BATCH_SIZE` | `32` | Chunks per embed call |
| `BATCH_SIZE` | `100` | DB operations per batch |

#### QA Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CTX_TOKENS` | `2800` | Context token budget |
| `MAX_OUTPUT_TOKENS` | `192` | Response token limit |
| `PER_CHUNK_CHAR_CAP` | `900` | Max chars per chunk |
| `FOCUS_WINDOW` | `2` | Neighbor window size |
| `PER_DOC_MAX` | `3` | Max chunks per document |

---

## API Reference

### CLI Commands

```bash
# Ingest documents
python -m graph_rag ingest /path/to/documents [--chunk-size 1800] [--overlap 200]

# Ask a question
python -m graph_rag ask -q "What is the policy?" [--max-results 6] [--show-sources]

# System check
python -m graph_rag check

# Setup commands
python -m graph_rag setup init [--force]
python -m graph_rag setup verify
python -m graph_rag setup start [--chat-only] [--embed-only]
python -m graph_rag setup status

# Help
python -m graph_rag help [COMMAND] [--all]
```

### Python API

```python
from graph_rag.config import get_driver
from graph_rag.ingest.files import ingest_folder
from graph_rag.qa.answer import ask

# Initialize
driver = get_driver()

# Ingest documents
n_docs, n_chunks = ingest_folder(driver, "/path/to/docs")

# Ask a question
result = ask(driver, "What are the key policies?", top_k=6)
print(result["answer"])
print(result["citations"])

# Cleanup
from graph_rag.config import close_driver
close_driver()
```

### Return Types

#### `ingest_folder()` Returns

```python
Tuple[int, int]  # (documents_ingested, chunks_with_embeddings)
```

#### `ask()` Returns

```python
{
    "answer": str,           # Generated response
    "citations": List[str],  # Citation tags used
    "used_chunks": List[Dict], # Chunks in context
    "error": Optional[str]   # Error message if failed
}
```

---

## Deployment

### Prerequisites

- Python 3.8+ (3.9+ recommended)
- CUDA Toolkit 12.1+
- NVIDIA GPU (RTX 20/30/40 series)
- Neo4j 5.20+
- 16GB+ RAM recommended

### Installation

```bash
# Clone repository
git clone <repo-url>
cd graph-rag-neo4j

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify CUDA support
python -c "import llama_cpp; print('CUDA:', llama_cpp.llama_supports_gpu_offload())"
```

### Server Configuration

Edit `config/server_config.json`:

```json
{
    "embedding_server": {
        "model_path": "/path/to/qwen3-embed-0.6b.gguf",
        "port": 8080,
        "n_gpu_layers": -1
    },
    "chat_server": {
        "model_path": "/path/to/llama-2-7b-chat.gguf",
        "port": 8081,
        "n_gpu_layers": -1
    }
}
```

### Production Considerations

1. **Scaling**: Deploy Neo4j cluster for high availability
2. **Caching**: Consider Redis instead of SQLite for distributed cache
3. **Load Balancing**: Multiple embedding/chat servers behind proxy
4. **Monitoring**: Enable `ENABLE_PERFORMANCE_MONITORING=1`
5. **Security**: Use TLS for Neo4j (`NEO4J_ENCRYPTED=1`)

---

## License

See LICENSE file in repository root.

---

*Documentation generated for Graph RAG Neo4j v1.0*
