"""Brute-force normalized vector index with cosine search."""
import math


def _cosine(a, b):
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


class VectorIndex:
    def __init__(self, dim):
        self.dim = dim
        self.vectors = {}   # id -> list[float]
        self.meta = {}

    def add(self, vid, vec, meta=None):
        self.vectors[vid] = vec
        self.meta[vid] = meta or {}

    def remove(self, vid):
        self.vectors.pop(vid, None)
        self.meta.pop(vid, None)

    def search(self, query, top_k=10):
        scored = [(vid, _cosine(query, vec)) for vid, vec in self.vectors.items()]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def __len__(self):
        return len(self.vectors)
