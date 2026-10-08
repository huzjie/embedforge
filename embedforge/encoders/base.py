"""Base encoder interface.

Every encoder turns raw content of its modality into a sequence of hidden
vectors (list[list[float]], shape (n_tokens, hidden)). The unified model then
mean-pools to a single vector (pooled embedding) OR keeps the sequence for late
interaction retrieval.
"""
from ..utils.stable import stable_vector


class BaseEncoder:
    modality = "base"

    def __init__(self, cfg):
        self.cfg = cfg

    def encode(self, content):
        """Return a single hidden vector (list[float]) summarizing the content."""
        seq = self.encode_tokens(content)
        if not seq:
            return stable_vector(f"{self.modality}:empty", self.cfg.hidden)
        n = len(seq)
        d = len(seq[0])
        return [sum(r[i] for r in seq) / n for i in range(d)]

    def encode_tokens(self, content):
        """Return a sequence of hidden vectors (list[list[float]])."""
        raise NotImplementedError
