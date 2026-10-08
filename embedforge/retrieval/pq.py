"""Product Quantization (PQ): compress embeddings for memory-efficient search.

The reusable idea: split a d-dim vector into m sub-vectors, quantize each
sub-vector to the nearest centroid (k-means-ish via deterministic hashing in
this reference), and store only the centroid ids. Search is approximate but
uses ~log2(k) * d bits per vector instead of 32 * d.
"""
import math


class ProductQuantizer:
    def __init__(self, dim, m=8, k=256):
        self.dim = dim
        self.m = m
        self.k = k
        self.sub = math.ceil(dim / m)
        self.centroids = self._make_centroids()

    def _make_centroids(self):
        from ..utils.stable import stable_vector
        centroids = []
        for s in range(self.m):
            centroids.append([stable_vector(f"pq:{s}:{c}", self.sub, -1.0, 1.0) for c in range(self.k)])
        return centroids

    def _subvectors(self, vec):
        out = []
        for s in range(self.m):
            chunk = vec[s * self.sub:(s + 1) * self.sub]
            if len(chunk) < self.sub:
                chunk = chunk + [0.0] * (self.sub - len(chunk))
            out.append(chunk)
        return out

    def encode(self, vec):
        """Return list of centroid ids (one per sub-vector)."""
        from .index import _cosine
        codes = []
        for s, sv in enumerate(self._subvectors(vec)):
            best = max(range(self.k), key=lambda c: _cosine(sv, self.centroids[s][c]))
            codes.append(best)
        return codes

    def decode(self, codes):
        out = []
        for s, c in enumerate(codes):
            out.extend(self.centroids[s][c])
        return out[:self.dim]

    def bytes_per_vector(self):
        import math as _m
        bits = self.m * _m.ceil(_m.log2(self.k))
        return bits / 8.0


class PQIndex:
    def __init__(self, dim, m=8, k=256):
        self.pq = ProductQuantizer(dim, m, k)
        self.codes = {}   # id -> list[int]
        self.vectors = {}  # id -> original (for re-ranking)

    def add(self, vid, vec):
        self.vectors[vid] = vec
        self.codes[vid] = self.pq.encode(vec)

    def search(self, query, top_k=10, rerank=True):
        from .index import _cosine
        qcodes = self.pq.encode(query)
        scored = []
        for vid, codes in self.codes.items():
            approx = self.pq.decode(codes)
            d = _cosine(query, approx)
            scored.append((vid, d))
        scored.sort(key=lambda x: -x[1])
        # re-rank the top candidates against the original vectors
        if rerank:
            cand = scored[: max(top_k * 4, top_k)]
            cand = [(vid, _cosine(query, self.vectors[vid])) for vid, _ in cand]
            cand.sort(key=lambda x: -x[1])
            return cand[:top_k]
        return scored[:top_k]

    def __len__(self):
        return len(self.codes)
