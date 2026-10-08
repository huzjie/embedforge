"""ColBERTIndex: store per-token vectors per document and retrieve by MaxSim."""
from .maxsim import MaxSim


class ColBERTIndex:
    def __init__(self):
        self.docs = {}      # doc_id -> list[list[float]] (per-token vectors)
        self.meta = {}      # doc_id -> metadata dict

    def add(self, doc_id, tokens, meta=None):
        self.docs[doc_id] = tokens
        self.meta[doc_id] = meta or {}

    def remove(self, doc_id):
        self.docs.pop(doc_id, None)
        self.meta.pop(doc_id, None)

    def search(self, query_tokens, top_k=10):
        scored = MaxSim.score_against_index(query_tokens, self.docs)
        return scored[:top_k]

    def __len__(self):
        return len(self.docs)
