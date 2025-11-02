"""Create a separate virtual environment for running the llama_cpp server and install server deps.

Usage (PowerShell):
    python setup_server_env.py

Creates: .venv-server
Installs packages from requirements-server.txt
"""
from __future__ import annotations
import pathlib
import subprocess
import sys
import shutil
import venv

ROOT = pathlib.Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv-server"
REQ_FILE = ROOT / "requirements-server.txt"


def run(cmd):
    print("==>", " ".join(cmd))
    subprocess.check_call(cmd)


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
    if sys.platform.startswith("win"):
        return venv_dir / "Scripts" / "python.exe"
    else:
        return venv_dir / "bin" / "python"


def install_requirements(venv_dir: pathlib.Path, req_file: pathlib.Path):
    py = str(venv_python(venv_dir))
    run([py, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run([py, "-m", "pip", "install", "-r", str(req_file)])


def main():
    if not REQ_FILE.exists():
        print(f"Missing {REQ_FILE}. Create it first.")
        sys.exit(1)
    created = create_venv(VENV_DIR)
    if created or VENV_DIR.exists():
        try:
            install_requirements(VENV_DIR, REQ_FILE)
        except subprocess.CalledProcessError as e:
            print("Installation failed:", e)
            sys.exit(2)
        print("Server venv ready.")
        print(r"Activate in PowerShell: .\.venv-server\Scripts\Activate.ps1")
    else:
        print("No venv created.")

if __name__ == '__main__':
    main()
