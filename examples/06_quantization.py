"""Quantize embeddings INT8/INT4 and measure retention + memory."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embedforge.config import load_config
from embedforge.model import UnifiedEmbedModel
from embedforge.quant import quantize_int8, dequantize_int8, quantize_int4, dequantize_int4, MemoryTracker


def main():
    cfg = load_config()
    model = UnifiedEmbedModel(cfg.model)
    v = model.embed("text", "quantize me")
    q8, s8 = quantize_int8(v)
    d8 = dequantize_int8(q8, s8)
    q4, sc4 = quantize_int4(v)
    d4 = dequantize_int4(q4, sc4)
    from embedforge.bench.quant import _cos, _mse
    print(f"int8: mse={_mse(v, d8):.6f}  cosine={_cos(v, d8):.4f}")
    print(f"int4: mse={_mse(v, d4):.6f}  cosine={_cos(v, d4):.4f}")
    print()
    mt = MemoryTracker(cfg.model.dim, cfg.model.hidden, cfg.model.vocab, cfg.model.n_layers)
    print(mt.report())


if __name__ == "__main__":
    main()
