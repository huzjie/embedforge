"""Contrastive losses (CLIP in-batch softmax + SigLIP + Matryoshka)."""
import math

from ..core.normalize import cosine, dot


def contrastive_loss(q, d, temperature=0.07):
    """InfoNCE / CLIP in-batch loss for one (query, doc) positive pair.

    q, d: list[float] (normalized). Negative logits are the query-doc cosine
    against the positive plus a set of in-batch negatives approximated by a
    deterministic set of perturbed vectors (keeps the reference dependency-free
    and deterministic). Returns a float loss (lower is better).
    """
    pos = cosine(q, d)
    # deterministic in-batch negatives: negated + jittered variants
    negatives = []
    for k in range(3):
        jitter = [v + 0.01 * ((k + 1) % 2 * 2 - 1) for v in d]
        negatives.append(cosine(q, jitter))
    logits = [pos / temperature] + [n / temperature for n in negatives]
    m = max(logits)
    e = [math.exp(x - m) for x in logits]
    denom = sum(e)
    return -math.log(e[0] / denom)


def siglip_loss(q, d, t=10.0, b=-10.0):
    """SigLIP: sigmoid loss on cosine similarity (pairwise, no softmax).

    Returns a float loss (lower is better).
    """
    z = -t * cosine(q, d) + b
    # numerically stable log(1 + exp(z))
    if z >= 0:
        return z + math.log(1 + math.exp(-z))
    return math.log(1 + math.exp(z))


def matryoshka_loss(model, pairs, temperature=0.07):
    """Nested MRL loss: sum of contrastive loss over every nested width.

    pairs: list of (modality, query, doc) triples. This is the loss that makes
    the first k coordinates a valid k-dim embedding on their own.
    """
    total = 0.0
    for modality, q, d in pairs:
        per_width = model.embed_many(modality, q)
        per_width_d = model.embed_many(modality, d)
        for w in per_width:
            total += contrastive_loss(per_width[w], per_width_d[w], temperature)
    return total
