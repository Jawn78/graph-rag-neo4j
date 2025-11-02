import json
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional, List, Any
import requests
from dataclasses import dataclass

@dataclass
class ServerConfig:
    model_path: str
    host: str
    port: int
    n_gpu_layers: int
    n_threads: int
    verbose: bool = True
    chat_format: Optional[str] = None
    n_ctx: Optional[int] = None
    n_batch: Optional[int] = None
    model_alias: Optional[str] = None
    embedding: bool = False

class ServerManager:
    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "server_config.json"
        self.config_path = config_path
        self.servers: Dict[str, subprocess.Popen] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load server configuration from JSON file."""
        with open(self.config_path) as f:
            config = json.load(f)
            
        self.chat_config = ServerConfig(**config["chat_server"])
        self.embedding_config = ServerConfig(**config["embedding_server"])

    def start_servers(self) -> Dict[str, bool]:
        """Start both chat and embedding servers."""
        results = {}
        results["chat"] = self.start_chat_server()
        results["embedding"] = self.start_embedding_server()
        return results

    def start_chat_server(self) -> bool:
        """Start the chat server."""
        cmd = [
            "python", "-m", "llama_cpp.server",
            "--model", self.chat_config.model_path,
            "--host", self.chat_config.host,
            "--port", str(self.chat_config.port),
            "--n_gpu_layers", str(self.chat_config.n_gpu_layers),
            "--n_threads", str(self.chat_config.n_threads),
            "--chat_format", self.chat_config.chat_format,
            "--n_ctx", str(self.chat_config.n_ctx),
            "--n_batch", str(self.chat_config.n_batch)
        ]
        if self.chat_config.verbose:
            cmd.extend(["--verbose", "true"])

        self.servers["chat"] = subprocess.Popen(cmd)
        return self.check_server_health("chat")

    def start_embedding_server(self) -> bool:
        """Start the embedding server."""
        cmd = [
            "python", "-m", "llama_cpp.server",
            "--model", self.embedding_config.model_path,
            "--host", self.embedding_config.host,
            "--port", str(self.embedding_config.port),
            "--model_alias", self.embedding_config.model_alias,
            "--n_gpu_layers", str(self.embedding_config.n_gpu_layers),
            "--n_threads", str(self.embedding_config.n_threads)
        ]
        if self.embedding_config.embedding:
            cmd.extend(["--embedding", "true"])
        if self.embedding_config.verbose:
            cmd.extend(["--verbose", "true"])

        self.servers["embedding"] = subprocess.Popen(cmd)
        return self.check_server_health("embedding")

    def check_server_health(self, server_type: str, max_retries: int = 5) -> bool:
        """Check if a server is healthy."""
        config = self.chat_config if server_type == "chat" else self.embedding_config
        url = f"http://{config.host}:{config.port}/health"
        
        for _ in range(max_retries):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    return True
                time.sleep(2)
            except requests.RequestException:
                time.sleep(2)
                continue
        return False

    def stop_servers(self) -> None:
        """Stop all running servers."""
        for server in self.servers.values():
            server.terminate()
            server.wait()
        self.servers.clear()

    def get_server_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all servers."""
        status = {}
        for server_type in ["chat", "embedding"]:
            config = self.chat_config if server_type == "chat" else self.embedding_config
            status[server_type] = {
                "running": server_type in self.servers and self.servers[server_type].poll() is None,
                "healthy": self.check_server_health(server_type),
                "port": config.port,
                "gpu_layers": config.n_gpu_layers
            }
        return status

    def __enter__(self):
        """Context manager support for auto-starting servers."""
        self.start_servers()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support for auto-stopping servers."""
        self.stop_servers()