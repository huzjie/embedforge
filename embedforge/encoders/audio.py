"""Audio encoder: samples -> spectrogram frames -> linear projection -> sequence.

The reusable idea: raw audio is mapped to log-mel-ish spectrogram frames, each
frame is linearly projected into the shared token space, then mean-pooled.
"""
from ..core.nn import Linear, LayerNorm
from ..utils.stable import stable_vector
from .base import BaseEncoder


class AudioEncoder(BaseEncoder):
    modality = "audio"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.proj = Linear(1, cfg.hidden, seed=3)
        self.layers = [
            (Linear(cfg.hidden, cfg.hidden, seed=30 + i), LayerNorm(cfg.hidden))
            for i in range(cfg.n_layers)
        ]

    def _frames(self, samples):
        n = self.cfg.audio_frames
        if isinstance(samples, (list, tuple)):
            flat = [float(v) for v in samples]
        else:
            flat = stable_vector("audio:default", max(n, 8))
        if len(flat) < n:
            flat = flat + [0.0] * (n - len(flat))
        # simple "spectrogram": absolute value per frame (energy envelope)
        return [abs(v) for v in flat[:n]]

    def encode_tokens(self, content):
        frames = self._frames(content)
        hseq = [self.proj([f]) for f in frames]
        for linear, ln in self.layers:
            hseq = [ln(linear(row)) for row in hseq]
        return hseq
