"""Unified model with Matryoshka (any-width) embeddings."""
from ..encoders import TextEncoder, ImageEncoder, AudioEncoder, CodeEncoder
from ..projection import MatryoshkaHead


class EmbedModelWithMRL:
    def __init__(self, cfg):
        self.cfg = cfg
        self.encoders = {
            "text": TextEncoder(cfg),
            "image": ImageEncoder(cfg),
            "audio": AudioEncoder(cfg),
            "code": CodeEncoder(cfg),
        }
        self.head = MatryoshkaHead(cfg.hidden, cfg.dim, cfg.mrl_dims, seed=5)

    def encode_hidden(self, modality, content):
        enc = self.encoders.get(modality)
        if enc is None:
            enc = self.encoders["text"]
        return enc.encode(content)

    def embed(self, modality, content, width=None):
        return self.head.embed(self.encode_hidden(modality, content), width)

    def embed_many(self, modality, content):
        """Return {width: embedding} for all nested widths."""
        return self.head.embed_many(self.encode_hidden(modality, content))

    def embed_tokens(self, modality, content):
        enc = self.encoders.get(modality)
        if enc is None:
            enc = self.encoders["text"]
        return enc.encode_tokens(content)

    @property
    def modalities(self):
        return sorted(self.encoders.keys())

    @property
    def mrl_dims(self):
        return list(self.head.mrl_dims)
