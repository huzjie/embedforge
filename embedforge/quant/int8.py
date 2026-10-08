"""INT8 absmax (per-tensor) quantization.

The reusable idea: find the max |value|, scale = 127 / max, round to nearest
int8. Dequantize reverses. This is the cheapest quantization and preserves
embedding quality well (cosine is scale-invariant after normalization).
"""


def _scale(x):
    m = max((abs(v) for v in x), default=1.0) or 1.0
    return 127.0 / m


def quantize_int8(vec):
    s = _scale(vec)
    return [max(-127, min(127, round(v * s))) for v in vec], s


def dequantize_int8(q, s):
    return [v / s for v in q]


def mse(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) / max(len(a), 1)
