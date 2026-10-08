"""End-to-end: encode -> index -> search -> quantize -> report."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embedforge.config import load_config
from embedforge.backends.cpu import CpuBackend
from embedforge.retrieval import VectorIndex
from embedforge.quant import quantize_int8, dequantize_int8, MemoryTracker


def main():
    cfg = load_config()
    cpu = CpuBackend(cfg)
    idx = VectorIndex(cfg.model.dim)
    docs = [f"queryable document {i}" for i in range(6)]
    for i, d in enumerate(docs):
        v = cpu.embed("text", d)
        q, s = quantize_int8(v)
        idx.add(f"doc-{i}", dequantize_int8(q, s), {"title": d})
    hits = idx.search(cpu.embed("text", "document 4"), top_k=3)
    print("top hits (quantized storage):")
    for vid, score in hits:
        print(f"  {score:.4f}  {idx.meta[vid]['title']}")
    print()
    mt = MemoryTracker(cfg.model.dim, cfg.model.hidden, cfg.model.vocab, cfg.model.n_layers)
    print(mt.report())


if __name__ == "__main__":
    main()
