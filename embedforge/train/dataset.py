"""Multimodal pair dataset."""
from ..utils.stable import stable_choice, stable_vector, stable_ints


class MultiModalPairs:
    """A small synthetic multimodal (query, doc) pair set for demo/training.

    Queries and docs are generated deterministically per modality so the
    training loop is reproducible. Each pair is (modality, query, doc) where
    query/doc share a semantic "topic id" -- the positive signal.
    """

    MODALITIES = ("text", "image", "audio", "code")

    def __init__(self, n=16, seed=0):
        self.n = n
        self.pairs = []
        for i in range(n):
            modality = self.MODALITIES[i % len(self.MODALITIES)]
            topic = i // len(self.MODALITIES)
            query = self._sample(modality, topic)
            doc = self._sample(modality, topic, shift=1)
            self.pairs.append((modality, query, doc))

    def _sample(self, modality, topic, shift=0):
        key = f"ds:{modality}:{topic}:{shift}"
        if modality == "text":
            return f"query about topic {topic} shift {shift} the quick brown fox"
        if modality == "code":
            return f"def fn_{topic}(x):\n    return x + {shift}"
        if modality == "audio":
            return stable_vector(key, 32, -1.0, 1.0)
        # image: 8x8 pixel grid
        return [stable_vector(f"{key}:r{r}", 8, -1.0, 1.0) for r in range(8)]

    def __len__(self):
        return len(self.pairs)

    def __iter__(self):
        return iter(self.pairs)
