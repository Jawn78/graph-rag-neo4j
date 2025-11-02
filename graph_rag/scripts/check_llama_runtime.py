import sys
import pathlib

try:
    import llama_cpp
    print("llama_cpp package file:", llama_cpp.__file__)
    libdir = pathlib.Path(llama_cpp.__file__).parent / "lib"
    print("lib dir:", libdir)
    if libdir.exists():
        files = sorted(libdir.glob("*"))
        print(f"found {len(files)} files:")
        for p in files:
            print(" -", p.name)
    else:
        print("lib dir does not exist")

    try:
        from llama_cpp import _ggml
        lib = getattr(_ggml, "libggml", None)
        print("_ggml.libggml:", lib)
        if lib is None:
            print("libggml not loaded or not present")
        else:
            potential = [
                "ggml_cuda_init",
                "ggml_cuda_available",
                "ggml_cuda_has_device",
                "ggml_metal_init",
                "ggml_cuda_has_context",
            ]
            for name in potential:
                try:
                    getattr(lib, name)
                    print("FOUND symbol:", name)
                except Exception:
                    pass
    except Exception as e:
        print("could not import llama_cpp._ggml:", e)

except Exception as e:
    print("import error:", e)
    sys.exit(2)

sys.exit(0)
