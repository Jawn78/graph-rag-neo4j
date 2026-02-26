import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

class EnvironmentManager:
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.cuda_path = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5")
        
    def setup_environment(self, force: bool = False) -> None:
        """Set up the Python environment with all required dependencies."""
        # Set CUDA build environment variables
        os.environ["CMAKE_ARGS"] = (
            "-DGGML_CUDA=on "
            "-DCMAKE_CUDA_ARCHITECTURES=75;80;86;89 "
            f'-DCUDA_TOOLKIT_ROOT_DIR="{self.cuda_path}"'
        )
        
        # Install requirements
        requirements_file = self.workspace_root / "requirements.txt"
        cmd = [sys.executable, "-m", "pip", "install"]
        if force:
            cmd.extend(["--force-reinstall", "--no-cache-dir"])
        cmd.extend(["-r", str(requirements_file)])
        subprocess.run(cmd, check=True)
        
    def verify_cuda(self) -> Tuple[bool, Optional[str]]:
        """Verify CUDA installation and capabilities."""
        try:
            # Try to get CUDA info from nvidia-smi first
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version,cuda_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True
            )
            if result.returncode == 0:
                driver_version, cuda_version = result.stdout.strip().split(", ")
                return True, f"CUDA {cuda_version} available with driver version {driver_version}"
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # Fallback to checking CUDA_PATH
        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path and os.path.exists(cuda_path):
            return True, f"CUDA installation found at {cuda_path}"
        
        return False, "CUDA installation not found"
            
    def verify_dependencies(self) -> Dict[str, bool]:
        """Verify all required dependencies are installed."""
        required_packages = [
            "llama_cpp_python",
            "fastapi",
            "uvicorn",
            "neo4j",
            "numpy",
            "diskcache"
        ]
        
        status = {}
        for package in required_packages:
            try:
                __import__(package)
                status[package] = True
            except ImportError:
                status[package] = False
        return status

def setup_environment(force: bool = False):
    """Quick setup function for environment configuration."""
    workspace_root = Path(__file__).parent.parent.parent
    env_manager = EnvironmentManager(workspace_root)
    env_manager.setup_environment(force=force)

def verify_environment() -> Dict[str, Any]:
    """Verify the complete environment setup."""
    workspace_root = Path(__file__).parent.parent.parent
    env_manager = EnvironmentManager(workspace_root)
    
    cuda_status, cuda_info = env_manager.verify_cuda()
    deps_status = env_manager.verify_dependencies()
    
    return {
        "cuda": {
            "status": cuda_status,
            "info": cuda_info
        },
        "dependencies": deps_status,
        "environment": {
            "python": sys.version,
            "platform": sys.platform
        }
    }