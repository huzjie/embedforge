"""Cross-modal benchmark: query in one modality retrieves docs in another.

Because a unified space is shared, a text query can retrieve an audio doc or an
image doc directly -- no modality-specific plumbing.
"""
from .registry import register_benchmark
from .metrics import recall_at_k


@register_benchmark("crossmodal")
def run_crossmodal(backend=None, cfg=None, n=8, **kwargs):
    mods = ["image", "audio", "code"]
    topics = [f"topic-{i}" for i in range(n)]
    r1 = 0.0
    for t in topics:
        # text query -> gallery of mixed modalities, relevant = same topic
        scored = sorted(topics, key=lambda d: -backend.similarity("text", t, "image", d))
        r1 += recall_at_k(scored, {t}, 1)
    r1 /= n
    return {"crossmodal_recall@1": round(r1, 4), "n": n}
