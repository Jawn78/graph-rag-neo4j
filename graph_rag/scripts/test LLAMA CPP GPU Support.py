"""Check whether the installed llama.cpp (llama_cpp Python package) includes GPU support.

This script uses two heuristics:
 - Looks for library files in the package `lib/` directory whose filenames contain GPU-related keywords
   (cuda, metal, vulkan, mt, gpu).
 - Scans the DLL/shared object files for strings often present when built with CUDA/Metal/Vulkan support
   (e.g. 'cuda', 'cuInit', 'Metal', 'vulkan', 'vk') to detect compiled-in support even when filenames
   are generic.

Exit codes:
 - 0 : GPU support detected
 - 1 : No GPU support detected
 - 2 : Error (package missing or exception)

This is a best-effort check that avoids loading the model or making GPU runtime calls.
"""
from __future__ import annotations

import sys
import os
import pathlib
from typing import List

KEYWORDS = [b"cuda", b"CUDA", b"cuInit", b"cublas", b"nvrtc", b"nvml", b"metal", b"Metal", b"vulkan", b"vk", b"mt"]


def scan_file_for_keywords(path: pathlib.Path, keywords: List[bytes]) -> List[bytes]:
    found: List[bytes] = []
    try:
        with path.open("rb") as f:
            data = f.read()
    except Exception:
        return found
    for k in keywords:
        if k in data:
            found.append(k)
    return found


def check_llama_cpp_lib_dir() -> int:
    try:
        import llama_cpp
    except Exception as e:
        print("llama_cpp package not importable:", e)
        return 2

    pkg_dir = pathlib.Path(llama_cpp.__file__).parent
    lib_dir = pkg_dir / "lib"

    if not lib_dir.exists() or not lib_dir.is_dir():
        print(f"No lib/ directory found for llama_cpp at {lib_dir}")
        return 1

    dlls = list(lib_dir.glob("*"))
    if not dlls:
        print(f"No files found in {lib_dir}")
        return 1

    print(f"Inspecting {len(dlls)} file(s) in: {lib_dir}")

    # First-pass: filenames with gpu keywords
    filename_matches = []
    for p in dlls:
        name = p.name.lower()
        if any(k.decode('ascii').lower() in name for k in KEYWORDS if k.isascii()):
            filename_matches.append(p)

    if filename_matches:
        print("Found candidate library filenames that suggest GPU support:")
        for p in filename_matches:
            print(" -", p.name)
        return 0

    # Second-pass: scan binary contents for GPU-related strings
    binary_matches = {}
    for p in dlls:
        matches = scan_file_for_keywords(p, KEYWORDS)
        if matches:
            binary_matches[p] = matches

    if binary_matches:
        print("Found GPU-related strings inside library files (likely GPU support):")
        for p, ms in binary_matches.items():
            ms_str = ", ".join(m.decode('ascii', errors='ignore') for m in ms)
            print(f" - {p.name}: {ms_str}")
        return 0

    # Third-pass: try introspecting loaded ggml object (if available) for typical symbols
    try:
        # Try to import the internal _ggml module (if it exists) and inspect the loaded CDLL
        try:
            from llama_cpp import _ggml
        except Exception:
            _ggml = None

        if _ggml is not None:
            lib = getattr(_ggml, "libggml", None)
            if lib is not None:
                # try to access a handful of commonly-named CUDA/Metal symbols; getattr will raise
                potential_names = [
                    "ggml_cuda_init",
                    "ggml_cuda_available",
                    "ggml_cuda_has_device",
                    "ggml_metal_init",
                ]
                for nm in potential_names:
                    try:
                        getattr(lib, nm)
                        print(f"Detected symbol '{nm}' in loaded ggml library: likely GPU support")
                        return 0
                    except Exception:
                        continue
    except Exception:
        # ignore introspection failures; we've already done binary checks
        pass

    print("No GPU support indicators found in llama_cpp package's native libraries.")
    return 1


if __name__ == "__main__":
    code = check_llama_cpp_lib_dir()
    # Normal python exit() codes: 0 success, non-zero otherwise
    sys.exit(code)
