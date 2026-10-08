"""Neural building blocks (pure Python)."""
import math
import random


def _rng(seed):
    return random.Random(seed)


class Embedding:
    """Token embedding table (byte-level for text/code)."""

    def __init__(self, num, dim, seed=0):
        self.num = num
        self.dim = dim
        r = _rng(seed)
        self.table = [[r.uniform(-0.1, 0.1) for _ in range(dim)] for _ in range(num)]

    def __call__(self, ids):
        return [self.table[i % self.num] for i in ids]

    def lookup(self, i):
        return self.table[i % self.num]


class Linear:
    def __init__(self, in_dim, out_dim, seed=0):
        r = _rng(seed)
        self.w = [[r.uniform(-0.1, 0.1) for _ in range(out_dim)] for _ in range(in_dim)]
        self.b = [r.uniform(-0.1, 0.1) for _ in range(out_dim)]

    def __call__(self, x):
        out = [self.b[j] for j in range(len(self.b))]
        for i, xi in enumerate(x):
            for j in range(len(self.b)):
                out[j] += xi * self.w[i][j]
        return out


class LayerNorm:
    """Supports 1D vectors and 2D token sequences."""

    def __init__(self, dim, eps=1e-5):
        self.dim = dim
        self.eps = eps

    def __call__(self, x):
        if x and isinstance(x[0], (list, tuple)):
            return [self(row) for row in x]
        mu = sum(x) / len(x)
        var = sum((v - mu) ** 2 for v in x) / len(x)
        return [(v - mu) / (var + self.eps) ** 0.5 for v in x]


def scaled_dot_product(q, k, v, mask=None):
    """q/k/v: list[list[float]] (seq x dim). Returns weighted context vectors."""
    seq = len(q)
    d = len(q[0]) if q else 1
    scale = 1.0 / (d ** 0.5)
    out = []
    for i in range(seq):
        scores = [sum(q[i][dd] * k[j][dd] for dd in range(len(q[i]))) * scale for j in range(seq)]
        if mask is not None:
            for j in range(seq):
                if not mask[i][j]:
                    scores[j] = -1e9
        m = max(scores)
        e = [math.exp(x - m) for x in scores]
        s = sum(e) or 1e-9
        w = [x / s for x in e]
        ctx = [0.0] * len(v[0])
        for j in range(seq):
            for dd in range(len(v[0])):
                ctx[dd] += w[j] * v[j][dd]
        out.append(ctx)
    return out


def softmax(x):
    m = max(x)
    e = [math.exp(v - m) for v in x]
    s = sum(e) or 1e-9
    return [v / s for v in e]
