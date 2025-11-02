from pathlib import Path
import json
from typing import Dict, Any, Optional
import os
from pydantic import BaseModel, Field

# Neo4j configuration
NEO4J_HOST = os.getenv("NEO4J_HOST", "localhost")
NEO4J_PORT = int(os.getenv("NEO4J_PORT", "7687"))
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "password")
NEO4J_DB = os.getenv("NEO4J_DB", "neo4j")

def get_driver():
    """Get a Neo4j driver instance."""
    from neo4j import GraphDatabase
    return GraphDatabase.driver(
        f"bolt://{NEO4J_HOST}:{NEO4J_PORT}",
        auth=(NEO4J_USER, NEO4J_PASS)
    )

class ServerSettings(BaseModel):
    model_path: str
    host: str = "127.0.0.1"
    port: int
    n_gpu_layers: int
    n_threads: int = 8
    verbose: bool = True
    chat_format: Optional[str] = None
    n_ctx: Optional[int] = None
    n_batch: Optional[int] = None
    model_alias: Optional[str] = None
    embedding: bool = False

class Config:
    def __init__(self, config_dir: Optional[Path] = None) -> None:
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "config"
        self.config_dir = config_dir
        self.config_dir.mkdir(exist_ok=True)
        
    def load_server_config(self) -> Dict[str, ServerSettings]:
        """Load server configuration."""
        config_file = self.config_dir / "server_config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Server config not found at {config_file}")
            
        with open(config_file) as f:
            data = json.load(f)
        
        return {
            "chat": ServerSettings(**data["chat_server"]),
            "embedding": ServerSettings(**data["embedding_server"])
        }
        
    def save_server_config(self, config: Dict[str, ServerSettings]) -> None:
        """Save server configuration."""
        config_file = self.config_dir / "server_config.json"
        data = {
            "chat_server": config["chat"].dict(exclude_none=True),
            "embedding_server": config["embedding"].dict(exclude_none=True)
        }
        
        with open(config_file, 'w') as f:
            json.dump(data, f, indent=4)
            
    def get_cuda_config(self) -> Dict[str, Any]:
        """Get CUDA configuration."""
        return {
            "cuda_path": "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5",
            "architectures": [75, 80, 86, 89],  # RTX 20, 30, 40 series
            "build_flags": {
                "GGML_CUDA": "on",
                "CMAKE_CUDA_ARCHITECTURES": "75;80;86;89"
            }
        }