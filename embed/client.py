from openai import OpenAI
from ..config import EMBED_BASE_URL, EMBED_MODEL
_client = OpenAI(base_url=EMBED_BASE_URL, api_key="llamacpp")

def embed_one(text: str) -> list[float]:
    r = _client.embeddings.create(model=EMBED_MODEL, input=text, encoding_format="float")
    return r.data[0].embedding

def detect_dim() -> int:
    return len(embed_one("dimension probe"))
