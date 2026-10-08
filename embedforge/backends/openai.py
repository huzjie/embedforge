"""OpenAI-compatible embeddings backend (stdlib urllib, no requests dep).

Points at any OpenAI-compatible /v1/embeddings endpoint (OpenAI, Azure, or a
self-hosted vLLM/embedding server). Requires base_url + api_key + model in cfg.
"""
import base64
import json
import urllib.request

from .base import EmbeddingBackend
from .registry import register_backend


@register_backend("openai")
class OpenAIEmbeddingBackend(EmbeddingBackend):
    name = "openai"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.base_url = (cfg.backend.base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = cfg.backend.api_key
        self.model = cfg.backend.model or "text-embedding-3-small"

    def embed(self, modality, content):
        if not self.api_key:
            raise RuntimeError("openai backend requires backend.api_key")
        url = f"{self.base_url}/embeddings"
        payload = json.dumps({"model": self.model, "input": content}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        vec = data["data"][0]["embedding"]
        return vec
