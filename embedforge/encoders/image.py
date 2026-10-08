"""Image encoder: patchify -> linear projection -> ViT blocks -> sequence.

The reusable idea (from EmbeddingGemma 2 / ViT): an image is split into
non-overlapping patches, each patch is linearly projected into the token
embedding space, and the resulting sequence is processed by a transformer.
The same patch grid feeds the text-image contrastive head.
"""
from ..core.nn import Linear, LayerNorm, scaled_dot_product
from .base import BaseEncoder


class ImageEncoder(BaseEncoder):
    modality = "image"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.patch_dim = cfg.patch * cfg.patch
        self.proj = Linear(self.patch_dim, cfg.hidden, seed=2)
        self.layers = [
            (Linear(cfg.hidden, cfg.hidden, seed=20 + i), LayerNorm(cfg.hidden))
            for i in range(cfg.n_layers)
        ]

    def _pixels(self, content):
        # content: list of lists (H x W) of floats, a flat list, or a str
        h = self.cfg.img_h
        w = self.cfg.img_w
        if isinstance(content, str):
            from ..utils.stable import stable_vector
            flat = stable_vector(f"img:{content}", h * w, 0.0, 1.0)
        elif isinstance(content, (list, tuple)) and content and isinstance(content[0], (list, tuple)):
            flat = [float(v) for row in content for v in row]
        else:
            flat = [float(v) for v in content]
        # normalize to h*w length (pad / truncate)
        want = h * w
        if len(flat) < want:
            flat = flat + [0.0] * (want - len(flat))
        flat = flat[:want]
        return flat

    def encode_tokens(self, content):
        flat = self._pixels(content)
        p = self.cfg.patch
        h = self.cfg.img_h
        w = self.cfg.img_w
        patches = []
        for y in range(0, h, p):
            for x in range(0, w, p):
                patch = []
                for yy in range(y, min(y + p, h)):
                    for xx in range(x, min(x + p, w)):
                        patch.append(flat[yy * w + xx])
                # pad partial patch to patch_dim
                if len(patch) < self.patch_dim:
                    patch = patch + [0.0] * (self.patch_dim - len(patch))
                patches.append(patch)
        if not patches:
            patches = [[0.0] * self.patch_dim]
        hseq = [self.proj(patch) for patch in patches]
        for linear, ln in self.layers:
            hseq = scaled_dot_product(hseq, hseq, hseq)
            hseq = [ln(linear(row)) for row in hseq]
        return hseq
