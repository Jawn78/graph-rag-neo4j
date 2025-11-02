# Graph RAG

A Graph-based Retrieval Augmented Generation (RAG) system with Neo4j backend and GPU-accelerated LLM support.

## Features

- Document ingestion with configurable chunking
- Graph-based knowledge storage using Neo4j
- GPU-accelerated embedding and chat servers
- CLI interface for easy interaction
- CUDA support for optimal performance

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- Neo4j database
- PowerShell (for Windows users)

## Installation

1. Clone the repository
2. Create a virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

3. Initialize the environment:
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
