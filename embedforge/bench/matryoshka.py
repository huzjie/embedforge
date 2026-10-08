"""Matryoshka degradation benchmark: ranking agreement vs width.

Measures how well the top-k neighbor ranking at each nested width agrees with
the full-width ranking. A well-trained MRL model keeps high agreement even at
small widths (the first k coordinates remain informative on their own).
"""
from .registry import register_benchmark
from ..backends.cpu import CpuBackend


@register_benchmark("matryoshka")
def run_matryoshka(backend=None, cfg=None, n=12, k=5, **kwargs):
    from ..model import EmbedModelWithMRL
    model = EmbedModelWithMRL(cfg.model)
    texts = [f"document about subject {i} with enough words" for i in range(n)]
    full = {t: model.embed("text", t) for t in texts}
    full_order = {t: sorted(texts, key=lambda o: -_cos(full[t], full[o])) for t in texts}
    curve = {}
    for w in model.mrl_dims:
        emb = {t: model.embed("text", t, width=w) for t in texts}
        agree = 0.0
        for t in texts:
            order = sorted(texts, key=lambda o: -_cos(emb[t], emb[o]))
            agree += _overlap(order[:k], full_order[t][:k]) / k
        curve[str(w)] = round(agree / n, 4)
    return {"ranking_agreement_by_width": curve, "mrl_dims": model.mrl_dims}


def _cos(a, b):
    import math
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _overlap(a, b):
    return len(set(a) & set(b))
