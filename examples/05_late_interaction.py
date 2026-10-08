"""ColBERT-style late-interaction retrieval (per-token, no chunking)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embedforge.config import load_config
from embedforge.backends.cpu import CpuBackend
from embedforge.late import ColBERTIndex


def main():
    cfg = load_config()
    cpu = CpuBackend(cfg)
    docs = [
        "the quick brown fox jumps over the lazy dog",
        "a fast auburn fox leaps across a sleepy hound",
        "quantum entanglement in superconducting qubits",
    ]
    idx = ColBERTIndex()
    for i, d in enumerate(docs):
        idx.add(f"doc-{i}", cpu.embed_tokens("text", d), {"title": d})
    q_tokens = cpu.embed_tokens("text", "quick fox jumps dog")
    for did, score in idx.search(q_tokens, top_k=3):
        print(f"  {score:8.3f}  {idx.meta[did]['title']}")


if __name__ == "__main__":
    main()
