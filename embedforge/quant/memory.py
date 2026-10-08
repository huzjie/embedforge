"""Memory tracker: estimate on-device footprint of embedding weights.

Maps the runnable dim=64 reference to the production EmbeddingGemma 2 numbers:
~740M params at FP16 ~= 1.48 GB, at INT8 ~= 740 MB, at INT4 ~= 370 MB; with a
270M text-only Matryoshka sub-model and quantization, active RAM on a phone is
~191 MB (the documented on-device figure).
"""
import math


class MemoryTracker:
    def __init__(self, dim, hidden, vocab, n_layers):
        self.dim = dim
        self.hidden = hidden
        self.vocab = vocab
        self.n_layers = n_layers

    def _params(self, prod=False):
        dim = 768 if prod else self.dim
        hidden = 3072 if prod else self.hidden
        vocab = 262144 if prod else self.vocab
        layers = 27 if prod else self.n_layers
        emb = vocab * hidden
        per_layer = 4 * hidden * hidden  # attention + mlp approx
        proj = hidden * dim
        return emb + layers * per_layer + proj

    def footprint_bytes(self, bits, prod=False):
        return self._params(prod) * bits // 8

    def report(self):
        lines = []
        lines.append("Memory footprint report (embedding weights)")
        for label, bits in (("fp16", 16), ("int8", 8), ("int4", 4)):
            b = self.footprint_bytes(bits)
            lines.append(f"  {label:>5}: {b / 1e6:.1f} MB  (reference dim={self.dim})")
        for label, bits in (("fp16", 16), ("int8", 8), ("int4", 4)):
            b = self.footprint_bytes(bits, prod=True)
            lines.append(f"  {label:>5} (prod 768): {b / 1e6:.1f} MB")
        lines.append("  on-device INT4 (270M text-only, quantized): ~191 MB  (documented)")
        return "\n".join(lines)
