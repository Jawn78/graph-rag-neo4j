"""Create a local virtual environment and install pinned dependencies.

Usage (PowerShell):
    python setup_env.py

What it does:
- Creates a virtual environment at .venv (unless it already exists).
- Upgrades pip, setuptools, wheel inside the venv.
- Installs packages from requirements-pinned.txt using the venv pip.
- Performs quick import/version checks and lists llama_cpp native libs.

Note: This script will perform network installs. Run in a shell where you want the venv created.
"""
from __future__ import annotations

import os
import sys
import subprocess
import pathlib
import venv
import shutil

ROOT = pathlib.Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
REQ_FILE = ROOT / "requirements-pinned.txt"


def create_venv(venv_dir: pathlib.Path):
    if venv_dir.exists():
        print(f"Virtual environment already exists at {venv_dir}")
        answer = input("Remove and recreate it? [y/N]: ").strip().lower()
        if answer != "y":
            print("Leaving existing venv intact.")
            return False
        print("Removing existing venv...")
        shutil.rmtree(venv_dir)

    print(f"Creating virtual environment at {venv_dir}...")
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    return True


def venv_python(venv_dir: pathlib.Path) -> pathlib.Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    else:
        return venv_dir / "bin" / "python"


def run(cmd: list, env=None, check=True):
    print("==>", " ".join(cmd))
    subprocess.check_call(cmd, env=env)


def install_requirements(venv_dir: pathlib.Path, req_file: pathlib.Path):
    py = str(venv_python(venv_dir))
    # Upgrade packaging tools first
    run([py, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    # Install pinned requirements
    run([py, "-m", "pip", "install", "-r", str(req_file)])


def quick_checks(venv_dir: pathlib.Path):
    py = str(venv_python(venv_dir))
    code = r"""
import sys
import importlib
import pathlib
print('Python executable:', sys.executable)
for pkg in ['llama_cpp', 'numpy', 'diskcache', 'jinja2', 'requests']:
    try:
        m = importlib.import_module(pkg)
        print(pkg, 'version=', getattr(m, '__version__', getattr(m, 'VERSION', 'unknown')))
    except Exception as e:
        print(pkg, 'import failed:', e)

try:
    import llama_cpp
    libdir = pathlib.Path(llama_cpp.__file__).parent / 'lib'
    print('llama_cpp lib dir:', libdir)
    if libdir.exists():
        for p in sorted(libdir.glob('*')):
            print(' -', p.name)
    else:
        print('lib dir not found')
except Exception as e:
    print('llama_cpp import or inspection failed:', e)
"""
    run([py, "-c", code])


def main():
    if not REQ_FILE.exists():
        print(f"Required file {REQ_FILE} not found. Create or modify requirements-pinned.txt first.")
        sys.exit(1)

    created = create_venv(VENV_DIR)
    if created or VENV_DIR.exists():
        try:
            install_requirements(VENV_DIR, REQ_FILE)
        except subprocess.CalledProcessError as e:
            print("Installation failed:", e)
            print("You can inspect the venv pip logs or try running the install command manually.")
            sys.exit(2)

        print("Running quick import/version checks...")
        try:
            quick_checks(VENV_DIR)
        except subprocess.CalledProcessError:
            print("Quick checks failed. See output above.")
            sys.exit(3)

        print("Setup complete. To activate the venv in PowerShell:")
        print(r"    .\.venv\Scripts\Activate.ps1")
        print("Then run your server command inside the activated environment.")
    else:
        print("No virtual environment created.")


if __name__ == '__main__':
    main()
