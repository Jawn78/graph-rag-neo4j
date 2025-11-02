Windows build instructions: compile ggml/llama.cpp with CUDA and use with llama-cpp-python

This document walks you through producing a CUDA-enabled build of `llama.cpp` (ggml) on Windows and wiring it into your `llama-cpp-python` venv.

Overview
- You must compile ggml/llama.cpp with CUDA/CUBLAS enabled.
- Then copy the produced DLL(s) into the `llama_cpp/lib/` folder of the Python package that your server uses, or rebuild `llama-cpp-python` from source to link against the CUDA build.

Prereqs (install before building)
- Visual Studio 2022 (Desktop development with C++) or Build Tools with MSVC x64 toolchain
- CMake >= 3.21
- Ninja (optional but recommended)
- CUDA Toolkit matching your GPU driver (e.g., 12.x or 11.x); ensure `nvcc --version` works and `nvidia-smi` shows GPU
- Git

Sanity checks (PowerShell)
Run these and confirm they work:

```powershell
nvidia-smi
nvcc --version
cmake --version
cl.exe   # Visual Studio developer prompt
```

High-level steps
1. Clone `llama.cpp` and build with CUDA
2. Copy the generated DLL(s) (ggml.dll, llama.dll, or cuda-named dlls) into your python venv's `site-packages/llama_cpp/lib/`
3. Reinstall `llama-cpp-python` (optional) or just run the server using the replaced DLLs

Example PowerShell build script (developer prompt recommended)

- Open "x64 Native Tools Command Prompt for VS 2022" or run the Developer PowerShell
- Edit the CMAKE_CUDA_ARCHITECTURES to the architectures your GPU supports (e.g., 75, 80, 86)

```powershell
# Clone repo
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Create build dir
mkdir build; cd build

# Configure CMake (example uses Ninja generator)
# The exact CMake options below vary by repo version. Check llama.cpp CMakeLists for the correct option names.
cmake .. -G "Ninja" `
  -DCMAKE_BUILD_TYPE=Release `
  -DGGML_CUDA=ON `
  -DUSE_CUBLAS=ON `
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86"

# Build
cmake --build . --config Release -j

# After build, the DLLs will be in the build directory (or subfolder). Example file names:
# ggml.dll  ggml-cuda.dll  llama.dll  mtmd.dll  (names vary by fork)

# Copy desirable DLLs into your python package lib folder (adjust path to your venv/site-packages)
$dst = "D:\graph_rag2\.venv-server\Lib\site-packages\llama_cpp\lib"
Copy-Item -Path .\ggml.dll -Destination $dst -Force
Copy-Item -Path .\llama.dll -Destination $dst -Force
# If there is a ggml-cuda.dll or similar, copy it as well
Copy-Item -Path .\ggml-cuda.dll -Destination $dst -Force

# Optional: reinstall llama-cpp-python from source so it binds against the new libs
# (This step may vary between python binding implementations.)
cd ..\..
git clone https://github.com/abetlen/llama-cpp-python.git
cd llama-cpp-python
# If the python package supports LlamaCMake options, set environment variables as documented
# Example (may vary):
$env:LLAMA_CPP_SHARED_LIB_DIR = "D:\graph_rag2\llama.cpp\build"
python -m pip install --upgrade --force-reinstall --no-cache-dir .
```

Notes & pitfalls
- The exact CMake option names (`GGML_CUDA`, `USE_CUBLAS`, etc.) can vary between forks; always check the `CMakeLists.txt` in the repo you cloned.
- The copy-replace approach (copy DLLs into `site-packages/llama_cpp/lib/`) is a practical shortcut. If the Python bindings expect to find specific filenames, keep a backup of the original DLLs before overwriting.
- Ensure your CUDA runtime and driver are compatible with the CUDA version you used to build.
- Even with GPU-compiled native libs, some quantized models (e.g., q6_K) are CPU-only. If your model is quantized, try a small FP16 model to confirm GPU works.

If you want, I can generate a PowerShell script (non-destructive — it will back up existing DLLs) that automates the steps above and prompts you for: CMake generator, CUDA archs, paths and whether to reinstall the Python package. Tell me and I'll add it to the repo as `build_llama_cuda.ps1`.
