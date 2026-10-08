"""Build a product-quantized index and run an approximate search."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embedforge.config import load_config
from embedforge.backends.cpu import CpuBackend
from embedforge.retrieval import PQIndex


def main():
    cfg = load_config()
    cpu = CpuBackend(cfg)
    idx = PQIndex(cfg.model.dim, m=8, k=16)
    docs = [f"document about subject {i}" for i in range(8)]
    for i, d in enumerate(docs):
        idx.add(f"doc-{i}", cpu.embed("text", d))
    print(f"bytes/vector (PQ) = {idx.pq.bytes_per_vector():.2f} vs fp32 {cfg.model.dim * 4} bytes")
    hits = idx.search(cpu.embed("text", "subject 3"), top_k=3)
    for vid, score in hits:
        print(f"  {score:.4f}  {vid}")


if __name__ == "__main__":
    main()
