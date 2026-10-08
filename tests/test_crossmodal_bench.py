import unittest

from embedforge.config import load_config
from embedforge.backends.mock import MockBackend
from embedforge.bench.crossmodal import run_crossmodal


class TestCrossModalBench(unittest.TestCase):
    def test_improves(self):
        cfg = load_config()
        mock = MockBackend(cfg)
        before = run_crossmodal(backend=mock, cfg=cfg)["crossmodal_recall@1"]
        mock.skill = 1.0
        after = run_crossmodal(backend=mock, cfg=cfg)["crossmodal_recall@1"]
        self.assertGreaterEqual(after, before)


if __name__ == "__main__":
    unittest.main()
