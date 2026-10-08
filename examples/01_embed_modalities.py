"""Embed one sample in each modality and print dims + cross-modal similarity."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embedforge.config import load_config
from embedforge.model import UnifiedEmbedModel
from embedforge.core.normalize import cosine


def main():
    cfg = load_config()
    model = UnifiedEmbedModel(cfg.model)
    samples = {
        "text": "a cat sitting on a mat",
        "image": [[0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4] for _ in range(8)],
        "audio": [0.5, -0.3, 0.2, 0.7, -0.1, 0.9, 0.0, 0.4] * 4,
        "code": "def hello():\n    return 42",
    }
    emb = {}
    for m, c in samples.items():
        v = model.embed(m, c)
        emb[m] = v
        print(f"{m:>6}: dim={len(v)}  head={[round(x, 3) for x in v[:4]]}")
    print()
    print("cross-modal cosine similarity (text as query):")
    for m in ("image", "audio", "code"):
        print(f"  text<->{m}: {cosine(emb['text'], emb[m]):+.4f}")


if __name__ == "__main__":
    main()
