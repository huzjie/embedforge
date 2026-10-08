"""Common embedding backend interface."""
from ..core.normalize import l2_norm


class EmbeddingBackend:
    name = "base"

    def __init__(self, cfg):
        self.cfg = cfg

    @property
    def dim(self):
        return self.cfg.model.dim

    def embed(self, modality, content):
        """Return a normalized embedding vector (list[float] of len dim)."""
        raise NotImplementedError

    def similarity(self, a_modality, a, b_modality, b):
        """Cosine similarity between two embedded items (default)."""
        from ..core.normalize import cosine
        return cosine(self.embed(a_modality, a), self.embed(b_modality, b))
