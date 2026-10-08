"""Multimodal retrieval benchmark: text query -> image doc recall@k.

With the mock backend, recall climbs from ~0.5 (skill=0.5) to 1.0 (skill=1.0),
demonstrating that a higher-quality embedding model directly raises retrieval.
"""
from .registry import register_benchmark
from .metrics import recall_at_k, mean_average_precision


@register_benchmark("multimodal_retrieval")
def run_multimodal_retrieval(backend=None, cfg=None, n=10, k=5, **kwargs):
    topics = [f"topic-{i}" for i in range(n)]
    rankings = []
    relevant = {}
    for i, t in enumerate(topics):
        # text query -> image gallery
        scored = sorted(topics, key=lambda d: -backend.similarity("text", t, "image", d))
        rankings.append([topics.index(d) for d in scored])
        relevant[i] = {i}
    r1 = sum(recall_at_k(rankings[i], relevant[i], 1) for i in range(n)) / n
    rk = sum(recall_at_k(rankings[i], relevant[i], k) for i in range(n)) / n
    mAP = mean_average_precision(rankings, relevant)
    return {"recall@1": round(r1, 4), f"recall@{k}": round(rk, 4), "mAP": round(mAP, 4), "n": n}
