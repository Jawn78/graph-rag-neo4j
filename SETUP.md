# Graph RAG2 Setup Guide

This guide will walk you through setting up the Graph RAG2 project with GPU support.

## Prerequisites

1. **CUDA Toolkit 12.1 or later**
   - Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
   - Ensure you have compatible NVIDIA drivers installed

2. **Visual Studio 2022**
   - Community Edition is sufficient
   - Install "Desktop development with C++" workload
   - Include MSVC v143 build tools

3. **Python 3.9 or later**
   - Download from [Python.org](https://www.python.org/downloads/)
   - Add Python to PATH during installation

4. **Neo4j Database**
   ```powershell
   winget install Neo4j.Neo4j-Community
   ```

5. **Git**
   - Download from [Git-SCM](https://git-scm.com/downloads)

## Initial Setup

1. **Clone the Repository**
   ```powershell
   git clone https://your-repo-url/graph_rag2.git
   cd graph_rag2
   ```

2. **Create Virtual Environment**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

## CUDA Support Setup

1. **Build llama.cpp with CUDA Support**
   ```powershell
   # Run as administrator in a developer PowerShell
   .\build_llama_cuda.ps1
   ```

2. **Install GPU-enabled Python Bindings**
   ```powershell
   pip install llama-cpp-python --force-reinstall --no-cache-dir --verbose --upgrade --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu121
   ```

## Model Setup

1. **Download Required Models**
   - Place your GGUF models in a designated folder
   - Update `config.py` with model paths if needed

2. **Environment Configuration**
   - Copy `.env.example` to `.env`
   - Configure your settings:
     ```ini
     NEO4J_URI=bolt://localhost:7687
     NEO4J_USER=neo4j
     NEO4J_PASS=your_password
     DOCS_DIR=path/to/your/docs
     EMBED_MODEL=qwen3-embed-0.6b
     CHAT_MODEL=llama-2-7b-chat
     ```

## Starting Services

1. **Start Neo4j Database**
   ```powershell
   .\scripts\start-neo4j.ps1
   ```

2. **Start Embedding Server**
   ```powershell
   python -m llama_cpp.server `
     --model "path/to/embedding/model.gguf" `
     --host 127.0.0.1 `
     --port 8080 `
     --model_alias qwen3-embed-0.6b `
     --embedding true `
     --n_gpu_layers -1 `
     --n_threads 8
   ```

3. **Start Chat Server**
   ```powershell
   python -m llama_cpp.server `
     --model "path/to/chat/model.gguf" `
     --host 127.0.0.1 `
     --port 8081 `
     --n_gpu_layers 32 `
     --n_threads 8 `
     --chat_format chatml
   ```

## Verifying Installation

1. **Test GPU Support**
   ```powershell
   python test_servers.py
   ```

2. **Verify Neo4j Connection**
   ```powershell
   python -c "from config import get_driver; driver = get_driver(); print('Connected!' if driver else 'Failed!')"
   ```

## Troubleshooting

### CUDA Issues

1. **No GPU Detected**
   - Verify CUDA installation: `nvcc --version`
   - Check GPU drivers: `nvidia-smi`
   - Ensure CUDA paths are in system PATH

2. **Build Failures**
   - Run `build_llama_cuda.ps1` with verbose output
   - Check Visual Studio installation
   - Verify CUDA compute capability matches your GPU

### Neo4j Issues

1. **Connection Failed**
   - Verify Neo4j is running: `.\scripts\start-neo4j.ps1`
   - Check credentials in `.env`
   - Try connecting via Neo4j Browser: http://localhost:7474

### Python Package Issues

1. **DLL Load Failed**
   - Rebuild llama.cpp with `build_llama_cuda.ps1`
   - Reinstall Python package with CUDA support
   - Check PATH includes CUDA bin directory

## Performance Optimization

- Adjust `n_gpu_layers` based on your GPU memory
- Configure `n_threads` for optimal CPU usage
- Use `BATCH_SIZE` in config.py for embedding optimization
- Monitor GPU memory with `nvidia-smi -l 1`