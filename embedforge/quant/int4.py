"""INT4 group-wise absmax quantization.

The reusable idea (from on-device LLM/embedding serving): split the vector into
groups of `group_size`, compute an absmax scale PER GROUP (not per tensor), and
round to 4 bits. Group-wise scales track outliers much better than a single
per-tensor scale, so quality is preserved while memory drops ~8x.
"""


def _scale_group(g):
    m = max((abs(v) for v in g), default=1.0) or 1.0
    return 7.0 / m


def quantize_int4(vec, group_size=128):
    q = []
    scales = []
    for i in range(0, len(vec), group_size):
        g = vec[i:i + group_size]
        s = _scale_group(g)
        scales.append(s)
        q.extend(max(-7, min(7, round(v * s))) for v in g)
    return q, scales


def dequantize_int4(q, scales, group_size=128):
    out = []
    gi = 0
    for i in range(0, len(q), group_size):
        s = scales[gi]
        gi += 1
        out.extend(v / s for v in q[i:i + group_size])
    return out


def mse(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) / max(len(a), 1)
