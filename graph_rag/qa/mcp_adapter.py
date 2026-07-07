"""Minimal MCP agent adapter.
This module provides a small helper `call_mcp_agent(messages, tools=None, agent_url=None)`
that posts messages to an agent gateway which mediates communication with the Learn MCP server.

The Learn MCP server is not a stable OpenAI-style API endpoint; it's intended to be driven by
an agent framework. This adapter is intentionally lightweight: it expects you to provide a
working agent gateway (MCP_AGENT_URL) that accepts a JSON payload like:

POST /agent/run
{ "messages": [{"role":"system","content":"..."}, ...], "tools": [...] }

and returns a JSON object with `content` in a predictable place. If your agent gateway
has a different contract, adjust the adapter accordingly.
"""
from typing import Any, Dict, List, Optional, Tuple
import requests


def call_mcp_agent(messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None,
                   agent_url: Optional[str] = None, timeout: float = 10.0) -> Tuple[bool, str]:
    """Call the MCP agent gateway.

    Returns (ok, content_or_error).
    """
    if not agent_url:
        return False, "No agent_url configured for MCP agent gateway"

    payload = {"messages": messages}
    if tools is not None:
        payload["tools"] = tools

    try:
        resp = requests.post(agent_url, json=payload, timeout=timeout)
    except Exception as e:
        return False, f"Network error calling agent gateway: {e!r}"

    try:
        j = resp.json()
    except Exception as e:
        return False, f"Agent gateway returned non-JSON response: {e!r} (status {resp.status_code})"

    # Normalize response: expect j.get('content') or j.get('result') or deeper structure
    content = None
    if isinstance(j, dict):
        content = j.get("content") or j.get("result") or j.get("reply")
        # Some gateways embed in {'response': {'content': '...'}}
        if content is None:
            r = j.get("response")
            if isinstance(r, dict):
                content = r.get("content") or r.get("result")
    if content is None:
        return False, f"Agent gateway returned unexpected JSON structure: {j!r}"

    return True, str(content)
