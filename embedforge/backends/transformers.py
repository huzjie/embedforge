"""Transformers backend: real HuggingFace EmbeddingGemma via `transformers`.

Reference integration. Requires the `transformers` package and a local or
HF-hub model id; falls back to the CPU encoder if unavailable so the rest of
the framework always runs.
"""
from .base import EmbeddingBackend
from .registry import register_backend


@register_backend("transformers")
class TransformersBackend(EmbeddingBackend):
    name = "transformers"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_id = cfg.backend.model or "google/embeddinggemma-2"
        self._pipe = None
        self._fallback = None
        try:
            from transformers import AutoModel, AutoTokenizer  # noqa
            self._available = True
        except Exception:
            self._available = False

    def _load(self):
        if self._pipe is None:
            from transformers import AutoModel, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.model_id)
            self._pipe = AutoModel.from_pretrained(self.model_id)
        return self._pipe

    def embed(self, modality, content):
        if not self._available:
            from .cpu import CpuBackend
            if self._fallback is None:
                self._fallback = CpuBackend(self.cfg)
            return self._fallback.embed(modality, content)
        model = self._load()
        inputs = self._tok(str(content), return_tensors="pt")
        outputs = model(**inputs)
        vec = outputs.last_hidden_state.mean(dim=1)[0].detach().tolist()
        return vec
