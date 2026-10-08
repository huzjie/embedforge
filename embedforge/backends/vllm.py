"""vLLM backend: OpenAI-compatible endpoint served by vLLM.

vLLM exposes the OpenAI embeddings API; this backend reuses the OpenAI client
pointed at a vLLM server (default http://localhost:8000/v1).
"""
from .openai import OpenAIEmbeddingBackend
from .registry import register_backend


@register_backend("vllm")
class VLLMBackend(OpenAIEmbeddingBackend):
    name = "vllm"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.base_url = (cfg.backend.base_url or "http://localhost:8000/v1").rstrip("/")
        self.model = cfg.backend.model or "EmbeddingGemma"
