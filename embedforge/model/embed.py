"""Unified multimodal embedding model.

Routes each modality to its encoder, mean-pools to a single hidden vector, then
projects through ONE shared head into the shared embedding space. This is the
core idea of EmbeddingGemma 2: text / code / image / audio / video all land in
the same space, so cross-modal retrieval is just cosine similarity.
"""
from ..encoders import TextEncoder, ImageEncoder, AudioEncoder, CodeEncoder
from ..projection import SharedProjectionHead


class UnifiedEmbedModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.encoders = {
            "text": TextEncoder(cfg),
            "image": ImageEncoder(cfg),
            "audio": AudioEncoder(cfg),
            "code": CodeEncoder(cfg),
        }
        self.head = SharedProjectionHead(cfg.hidden, cfg.dim, seed=5)

    def encode_hidden(self, modality, content):
        enc = self.encoders.get(modality)
        if enc is None:
            enc = self.encoders["text"]
        return enc.encode(content)

    def embed(self, modality, content):
        """Return a normalized embedding vector (list[float] of len dim)."""
        return self.head.project(self.encode_hidden(modality, content))

    def embed_tokens(self, modality, content):
        """Return per-token hidden vectors (for late interaction)."""
        enc = self.encoders.get(modality)
        if enc is None:
            enc = self.encoders["text"]
        return enc.encode_tokens(content)

    @property
    def modalities(self):
        return sorted(self.encoders.keys())
