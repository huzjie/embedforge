"""Cross-modal search: text<->image<->audio in ONE embedding space.

The whole point of a unified embedding space: the query and the documents can be
DIFFERENT modalities. Search an image gallery with text, or an audio library
with an image -- all cosine similarity, no modality-specific plumbing.
"""
import math


def _cosine(a, b):
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


class CrossModalSearch:
    def __init__(self, model):
        self.model = model
        self.gallery = []  # list of (modality, content, id, meta)

    def index(self, modality, content, vid, meta=None):
        emb = self.model.embed(modality, content)
        self.gallery.append((modality, content, vid, emb, meta or {}))

    def search(self, query_modality, query_content, top_k=10):
        q = self.model.embed(query_modality, query_content)
        scored = [
            (vid, _cosine(q, emb), modality, meta)
            for modality, content, vid, emb, meta in self.gallery
        ]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def __len__(self):
        return len(self.gallery)
