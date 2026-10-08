"""Code encoder: whitespace/symbol-aware tokenizer over a text encoder.

The reusable idea: code is tokenized at a slightly higher level than raw bytes
(splitting on identifiers / punctuation) but still shares the same transformer
backbone and embedding space as natural language -- so a code search query and
a code snippet land in the same space.
"""
from ..core.nn import Embedding, Linear, LayerNorm, scaled_dot_product
from .base import BaseEncoder


class CodeEncoder(BaseEncoder):
    modality = "code"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.embed = Embedding(cfg.vocab, cfg.hidden, seed=4)
        self.layers = [
            (Linear(cfg.hidden, cfg.hidden, seed=40 + i), LayerNorm(cfg.hidden))
            for i in range(cfg.n_layers)
        ]

    def tokenize(self, code):
        # crude identifier/symbol split; each token hashed to [0, vocab)
        toks = []
        cur = ""
        for ch in code:
            if ch.isalnum() or ch == "_":
                cur += ch
            else:
                if cur:
                    toks.append(cur)
                    cur = ""
                if not ch.isspace():
                    toks.append(ch)
        if cur:
            toks.append(cur)
        if not toks:
            toks = ["0"]
        return [abs(hash(t)) % self.cfg.vocab for t in toks]

    def encode_tokens(self, content):
        ids = self.tokenize(content)
        h = self.embed(ids)
        for linear, ln in self.layers:
            h = scaled_dot_product(h, h, h)
            h = [ln(linear(row)) for row in h]
        return h
