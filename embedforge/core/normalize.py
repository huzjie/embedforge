"""L2 normalization and cosine similarity helpers."""
import math


def l2_norm(x):
    n = math.sqrt(sum(v * v for v in x)) or 1e-9
    return [v / n for v in x]


def cosine(a, b):
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def mean_pool(rows):
    """Mean-pool a sequence of vectors (list[list[float]]) into one vector."""
    if not rows:
        return []
    n = len(rows)
    d = len(rows[0])
    return [sum(r[i] for r in rows) / n for i in range(d)]


def truncate(vec, d):
    """Matryoshka truncation: keep the first d coordinates."""
    return vec[:d]
