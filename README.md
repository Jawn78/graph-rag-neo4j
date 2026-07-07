# Graph RAG

A Graph-based Retrieval Augmented Generation (RAG) system with Neo4j backend and GPU-accelerated LLM support.

## Features

- Document ingestion with configurable chunking
- Graph-based knowledge storage using Neo4j
- GPU-accelerated embedding and chat servers
- CLI interface for easy interaction
- CUDA support for optimal performance

## Requirements

- Python 3.9+
- CUDA-capable GPU (recommended)
- Neo4j 5.x database
- PowerShell (for Windows users)

## Installation

1. Clone the repository
2. Create a virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

3. Install the package (with document-parsing extras):
```powershell
pip install -e ".[ingest]"
```

4. Configure credentials — copy `.env.example` to `.env` and set at least
   `NEO4J_PASS` (there is deliberately no default password).

5. Initialize the environment:
```powershell
python -m graph_rag setup init
```

## Usage

### Start Services
```powershell
# Start Neo4j server
.\scripts\start-neo4j.ps1

# Start RAG servers
python -m graph_rag setup start
```

### Ingest Documents
```powershell
python -m graph_rag ingest docs/*.pdf
```

### Ask Questions
```powershell
python -m graph_rag ask -q "What is RAG?"
```

### Check System Status
```powershell
python -m graph_rag check
```

## Commands

- `ingest`: Ingest documents into the graph database
- `ask`: Query the RAG system
- `check`: Verify system status (CUDA, servers, Neo4j)
- `setup init`: Initialize environment
- `setup start`: Start server components
- `setup verify`: Check dependencies
- `help`: Show command documentation

## Development

```bash
pip install -e ".[dev]"
ruff check graph_rag tests   # lint
pytest tests/ -q             # unit tests
```

Project layout: everything lives in the `graph_rag` package — `ingest/`
(parsing, chunking, embedding), `graph/` (Neo4j schema, upserts, hybrid
search), `qa/` (answer pipeline), `context_engine/` (intent, rewriting,
reranking, personalization, feedback), `setup/` (environment and llama.cpp
server management).
