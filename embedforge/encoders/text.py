"""Byte-level text encoder: tokenize -> embed -> transformer -> sequence."""
from ..core.nn import Embedding, Linear, LayerNorm, scaled_dot_product
from ..utils.stable import stable_ints
from .base import BaseEncoder


class TextEncoder(BaseEncoder):
    modality = "text"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.embed = Embedding(cfg.vocab, cfg.hidden, seed=1)
        self.layers = [
            (Linear(cfg.hidden, cfg.hidden, seed=10 + i), LayerNorm(cfg.hidden))
            for i in range(cfg.n_layers)
        ]

    def tokenize(self, text):
        # byte-level tokenizer: each byte -> id in [0, vocab)
        b = text.encode("utf-8", errors="replace")
        return [x % self.cfg.vocab for x in b]

    def encode_tokens(self, content):
        ids = self.tokenize(content)
        if not ids:
            ids = [0]
        h = self.embed(ids)
        for linear, ln in self.layers:
            h = scaled_dot_product(h, h, h)
            h = [ln(linear(row)) for row in h]
        return h
