"""Quantization benchmark: accuracy retention under INT8 / INT4."""
from .registry import register_benchmark
from ..quant import quantize_int8, dequantize_int8, quantize_int4, dequantize_int4
from ..backends.cpu import CpuBackend


@register_benchmark("quantization")
def run_quant(backend=None, cfg=None, n=16, **kwargs):
    from ..model import UnifiedEmbedModel
    model = UnifiedEmbedModel(cfg.model)
    vecs = [model.embed("text", f"quant probe {i}") for i in range(n)]
    mse8 = 0.0
    mse4 = 0.0
    cos8 = 0.0
    cos4 = 0.0
    for v in vecs:
        q8, s8 = quantize_int8(v)
        d8 = dequantize_int8(q8, s8)
        mse8 += _mse(v, d8)
        cos8 += _cos(v, d8)
        q4, sc4 = quantize_int4(v)
        d4 = dequantize_int4(q4, sc4)
        mse4 += _mse(v, d4)
        cos4 += _cos(v, d4)
    n = max(len(vecs), 1)
    return {
        "int8_mse": round(mse8 / n, 6),
        "int8_cosine_retention": round(cos8 / n, 4),
        "int4_mse": round(mse4 / n, 6),
        "int4_cosine_retention": round(cos4 / n, 4),
    }


def _mse(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) / max(len(a), 1)


def _cos(a, b):
    import math
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return sum(x * y for x, y in zip(a, b)) / (na * nb)
