"""Launch llama_cpp.server using the .venv-server Python, capture logs for a timeout, and report GPU-related lines.

Usage (from repo root):
  ./.venv-server/Scripts/python.exe run_server_probe.py

This script will:
 - spawn the server subprocess (using the .venv-server Python executable)
 - stream stdout/stderr and print lines to this process
 - search lines for GPU-related keywords and summarize findings
 - terminate the server after `timeout` seconds

Be careful: this will start and stop the server; it does not persist it.
"""
from __future__ import annotations

import subprocess
import sys
import time
import shlex
from pathlib import Path

VENV_PY = Path(__file__).resolve().parent / ".venv-server" / "Scripts" / "python.exe"
TIMEOUT = 15

# Default server command (use your command here if different)
SERVER_CMD = [
    str(VENV_PY),
    "-u",
    "-m",
    "llama_cpp.server",
    "--model",
    r"F:\Models\Qwen3-Embedding-8B-Q6_K.gguf",
    "--host",
    "127.0.0.1",
    "--port",
    "8080",
    "--model_alias",
    "qwen3-embed-0.6b",
    "--embedding",
    "true",
    "--n_gpu_layers",
    "-1",
    "--n_threads",
    "8",
    "--verbose",
    "true",
]

GPU_KEYWORDS = [
    "cuda",
    "cuInit",
    "cublas",
    "nvrtc",
    "nvml",
    "metal",
    "vulkan",
    "vk",
    "offload",
    "GPU",
    "ggml_cuda",
    "ggml_metal",
]


def main():
    if not VENV_PY.exists():
        print(f"Venv python not found at {VENV_PY}. Ensure .venv-server exists and is created.")
        sys.exit(2)

    print("Starting server subprocess:")
    print(" ".join(shlex.quote(p) for p in SERVER_CMD))

    p = subprocess.Popen(SERVER_CMD, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    start = time.time()
    found_lines = []
    try:
        # Read lines until timeout
        while True:
            if p.stdout is None:
                break
            line = p.stdout.readline()
            if not line:
                # no output right now
                if p.poll() is not None:
                    break
                time.sleep(0.1)
            else:
                try:
                    text = line.decode('utf-8', errors='replace').rstrip()
                except Exception:
                    text = repr(line)
                print(text)
                lower = text.lower()
                for kw in GPU_KEYWORDS:
                    if kw.lower() in lower:
                        found_lines.append((kw, text))

            if time.time() - start > TIMEOUT:
                print(f"Timeout {TIMEOUT}s reached, terminating server subprocess...")
                break
    finally:
        # Try graceful terminate
        try:
            p.terminate()
            # wait briefly
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        except Exception:
            pass

    print("--- probe summary ---")
    if found_lines:
        print("Detected GPU-related log lines:")
        for kw, txt in found_lines:
            print(f"[{kw}] {txt}")
        sys.exit(0)
    else:
        print("No GPU-related keywords detected in the first", TIMEOUT, "seconds of logs.")
        print("If you see that all layers were assigned to CPU in earlier logs, the build or the model quantization is likely the issue.")
        sys.exit(1)


if __name__ == '__main__':
    main()
