"""Deterministic trainable mock backend.

The embedding quality is controlled by `skill` in [0, 1]. A content's "gold"
semantic direction is deterministic; the mock mixes it with a noise direction
weighted by (1 - skill). As skill -> 1.0, embeddings of the SAME content (any
modality) become identical, so retrieval recall -> 1.0.

Reusable tricks baked in:
- signal/noise ratio: noise must span the full [0,1] range or recall saturates
  at 100% too early (skill=0.5 should give ~0.67 recall, not 1.0).
- training signal must be always-positive so skill climbs monotonically.
- gold key is the *content* (semantic topic), so cross-modal positives share
  one direction regardless of modality.
"""
from ..core.normalize import l2_norm
from ..utils.stable import stable_vector, stable_float
from .base import EmbeddingBackend
from .registry import register_backend


@register_backend("mock")
class MockBackend(EmbeddingBackend):
    name = "mock"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.skill = 0.5

    def embed(self, modality, content):
        gold = stable_vector(f"gold:{content}", self.dim, -1.0, 1.0)
        noise = stable_vector(f"noise:{modality}:{content}", self.dim, -1.0, 1.0)
        s = self.skill
        raw = [s * g + (1.0 - s) * n for g, n in zip(gold, noise)]
        return l2_norm(raw)

    def similarity(self, a_modality, a, b_modality, b):
        # skill-gated semantic similarity: how much of the "true" semantic match
        # (same content -> 1.0) leaks through vs pure retrieval noise in [0,1].
        # skill=0.5 -> recall@1 ~0.3 over a 10-doc gallery; skill=1.0 -> 1.0.
        same = 1.0 if a == b else 0.0
        noise = stable_float(f"sim:{a_modality}:{a}:{b_modality}:{b}", 0.0, 1.0)
        return self.skill * same + noise
