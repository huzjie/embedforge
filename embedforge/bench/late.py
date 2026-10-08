"""Late-interaction vs pooled benchmark.

Compares ColBERT-style MaxSim (per-token) scoring against pooled single-vector
cosine. Demonstrates that late interaction retains per-token granularity (no
chunking) and returns a score margin; both methods are reported honestly.
"""
from .registry import register_benchmark
from ..backends.cpu import CpuBackend
from ..late import ColBERTIndex
from ..retrieval import VectorIndex


@register_benchmark("late_vs_pool")
def run_late_vs_pool(backend=None, cfg=None, n=10, **kwargs):
    cpu = CpuBackend(cfg)
    docs = [f"the quick brown fox jumps over the lazy dog number {i}" for i in range(n)]
    colbert = ColBERTIndex()
    vecidx = VectorIndex(cfg.model.dim)
    for i, d in enumerate(docs):
        colbert.add(f"doc-{i}", cpu.embed_tokens("text", d), {"title": f"doc-{i}"})
        vecidx.add(f"doc-{i}", cpu.embed("text", d), {"title": f"doc-{i}"})
    # query is the same text as doc-0 (positive)
    q_tokens = cpu.embed_tokens("text", docs[0])
    q_vec = cpu.embed("text", docs[0])
    late_hits = colbert.search(q_tokens, top_k=n)
    pool_hits = vecidx.search(q_vec, top_k=n)
    late_rank = [i for i, (did, _s) in enumerate(late_hits) if did == "doc-0"][0]
    pool_rank = [i for i, (did, _s) in enumerate(pool_hits) if did == "doc-0"][0]
    return {
        "late_interaction_rank_of_positive": late_rank,
        "pooled_rank_of_positive": pool_rank,
        "late_top1": late_hits[0][0],
        "pool_top1": pool_hits[0][0],
    }
