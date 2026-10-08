"""RAG pipeline: retrieve + (optional) generation over the retrieved context.

A minimal retrieval-augmented generation loop: embed the query, retrieve the
top-k documents, assemble a context block, and (if a generator is supplied)
augment it. The mock generator returns a deterministic citation-aware answer.
"""
from .crossmodal import CrossModalSearch
from ..utils.stable import stable_float


class RAGPipeline:
    def __init__(self, model, index=None, generator=None):
        self.model = model
        self.index = index if index is not None else CrossModalSearch(model)
        self.generator = generator

    def retrieve(self, query, top_k=5, modality="text"):
        return self.index.search(modality, query, top_k=top_k)

    def _build_context(self, hits):
        parts = []
        for i, (vid, score, modality, meta) in enumerate(hits, 1):
            parts.append(f"[{i}] ({modality}) {meta.get('title', vid)} (score={score:.3f})")
        return "\n".join(parts)

    def answer(self, query, top_k=5):
        hits = self.retrieve(query, top_k=top_k)
        ctx = self._build_context(hits)
        if self.generator is None:
            return f"Retrieved {len(hits)} documents:\n{ctx}"
        return self.generator(query, ctx)

    @staticmethod
    def mock_generator(query, ctx):
        conf = stable_float(f"rag:{query}", 0.7, 0.98)
        return f"Answer grounded in:\n{ctx}\n(confidence {conf:.2f})"
