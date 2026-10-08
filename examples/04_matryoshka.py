"""Embed at every nested width from ONE model (no retraining)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embedforge.config import load_config
from embedforge.model import EmbedModelWithMRL


def main():
    cfg = load_config()
    model = EmbedModelWithMRL(cfg.model)
    print(f"one model serves widths: {model.mrl_dims}")
    for w in model.mrl_dims:
        v = model.embed("text", "matryoshka", width=w)
        print(f"  width={w:>3}: dim={len(v)}")


if __name__ == "__main__":
    main()
