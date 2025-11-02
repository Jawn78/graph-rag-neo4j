import requests
import time
from typing import Dict, Optional

def check_server_health(host: str, port: int, max_retries: int = 5, retry_delay: int = 2) -> Dict[str, bool]:
    """
    Check if the server is healthy by making a request to the health endpoint.
    
    Args:
        host: Server host
        port: Server port
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        Dict containing server status information
    """
    url = f"http://{host}:{port}/health"
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return {
                    "healthy": True,
                    "gpu_enabled": "cuda" in response.json().get("backend", "").lower(),
                }
            time.sleep(retry_delay)
        except requests.RequestException:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            continue
            
    return {"healthy": False, "gpu_enabled": False}

if __name__ == "__main__":
    # Check both servers
    chat_status = check_server_health("127.0.0.1", 8081)
    embed_status = check_server_health("127.0.0.1", 8080)
    
    print("Chat Server Status:", "✓" if chat_status["healthy"] else "✗")
    print("Chat Server GPU:", "✓" if chat_status["gpu_enabled"] else "✗")
    print("Embedding Server Status:", "✓" if embed_status["healthy"] else "✗")
    print("Embedding Server GPU:", "✓" if embed_status["gpu_enabled"] else "✗")