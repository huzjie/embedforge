"""Matryoshka Representation Learning (MRL) head.

The reusable idea behind EmbeddingGemma 2's "single checkpoint, any width":
train with a *nested* loss summed over several truncation widths, so the first
k coordinates of the embedding are themselves a good embedding of width k. At
inference you can therefore serve 64 / 128 / 256 / 768 dims from ONE model, no
retraining -- the front of the vector is valid on its own.
"""
from ..core.normalize import l2_norm, truncate


class MatryoshkaHead:
    def __init__(self, hidden, dim, mrl_dims=None, seed=0):
        self.hidden = hidden
        self.dim = dim
        self.mrl_dims = list(mrl_dims) if mrl_dims else [dim]
        # ensure the full dim is always the last nested width
        if dim not in self.mrl_dims:
            self.mrl_dims.append(dim)
        self.mrl_dims = sorted(set(self.mrl_dims))
        from .head import SharedProjectionHead
        self.head = SharedProjectionHead(hidden, dim, seed=seed)

    def embed(self, hidden_vec, width=None):
        full = self.head.project(hidden_vec)
        if width is None or width >= self.dim:
            return full
        return l2_norm(truncate(full, width))

    def embed_many(self, hidden_vec):
        """Return {width: embedding} for every nested width (for MRL loss)."""
        full = self.head.project(hidden_vec)
        out = {}
        for w in self.mrl_dims:
            if w >= self.dim:
                out[w] = full
            else:
                out[w] = l2_norm(truncate(full, w))
        return out

    def __call__(self, hidden_vec, width=None):
        return self.embed(hidden_vec, width)
