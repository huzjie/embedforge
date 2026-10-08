"""CPU backend: run the real (pure-Python) unified encoders."""
from ..model import UnifiedEmbedModel
from .base import EmbeddingBackend
from .registry import register_backend


@register_backend("cpu")
class CpuBackend(EmbeddingBackend):
    name = "cpu"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = UnifiedEmbedModel(cfg.model)

    def embed(self, modality, content):
        return self.model.embed(modality, content)

    def embed_tokens(self, modality, content):
        return self.model.embed_tokens(modality, content)
