"""Shared projection head: hidden -> embedding dim -> L2 norm."""
from ..core.nn import Linear
from ..core.normalize import l2_norm


class SharedProjectionHead:
    """One shared head maps every modality's hidden vector into the same dim."""

    def __init__(self, hidden, dim, seed=0):
        self.hidden = hidden
        self.dim = dim
        self.linear = Linear(hidden, dim, seed=seed)

    def project(self, hidden_vec):
        return l2_norm(self.linear(hidden_vec))

    def __call__(self, hidden_vec):
        return self.project(hidden_vec)
