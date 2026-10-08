"""Train the mock backend and watch retrieval recall climb 0.5 -> 1.0."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embedforge.config import load_config
from embedforge.backends.mock import MockBackend
from embedforge.train.trainer import MockTrainer
from embedforge.bench import run_all


def main():
    cfg = load_config()
    mock = MockBackend(cfg)
    print(f"skill={mock.skill:.3f}")
    before = run_all(backend=mock, cfg=cfg)["multimodal_retrieval"]
    print(f"before: {before}")
    trainer = MockTrainer(mock, lr=cfg.train.lr)
    trainer.train(steps=cfg.train.steps)
    print(f"skill={mock.skill:.3f}")
    after = run_all(backend=mock, cfg=cfg)["multimodal_retrieval"]
    print(f"after:  {after}")


if __name__ == "__main__":
    main()
