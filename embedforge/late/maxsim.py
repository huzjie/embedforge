"""MaxSim late-interaction scorer (ColBERT-style).

The reusable idea behind pplx-embed-v2-late / ColBERT: instead of pooling each
document into ONE vector, keep a vector PER TOKEN. Scoring is then the sum over
query tokens of the maximum cosine similarity against all document tokens --
so a query term can match any position in the document, and documents need no
OCR or pre-chunking.
"""
import math


def cosine(a, b):
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


class MaxSim:
    @staticmethod
    def score(query_tokens, doc_tokens):
        """Sum over query tokens of max cosine against all doc tokens."""
        if not query_tokens or not doc_tokens:
            return 0.0
        total = 0.0
        for qt in query_tokens:
            best = -1.0
            for dt in doc_tokens:
                c = cosine(qt, dt)
                if c > best:
                    best = c
            total += best
        return total

    @staticmethod
    def score_against_index(query_tokens, doc_token_store):
        """Score a query against many documents (returns list of (doc_id, score))."""
        out = []
        for doc_id, tokens in doc_token_store.items():
            out.append((doc_id, MaxSim.score(query_tokens, tokens)))
        out.sort(key=lambda x: -x[1])
        return out
