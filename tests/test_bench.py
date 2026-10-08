import unittest

from embedforge.config import load_config
from embedforge.backends.mock import MockBackend
from embedforge.bench import list_benchmarks, run_all


class TestBench(unittest.TestCase):
    def test_benchmarks_registered(self):
        names = list_benchmarks()
        self.assertIn("multimodal_retrieval", names)
        self.assertIn("crossmodal", names)
        self.assertIn("matryoshka", names)

    def test_run_all(self):
        cfg = load_config()
        mock = MockBackend(cfg)
        results = run_all(backend=mock, cfg=cfg)
        self.assertIn("multimodal_retrieval", results)
        self.assertIn("recall@1", results["multimodal_retrieval"])

    def test_recall_improves(self):
        cfg = load_config()
        mock = MockBackend(cfg)
        before = run_all(backend=mock, cfg=cfg)["multimodal_retrieval"]["recall@1"]
        mock.skill = 1.0
        after = run_all(backend=mock, cfg=cfg)["multimodal_retrieval"]["recall@1"]
        self.assertGreaterEqual(after, before)


if __name__ == "__main__":
    unittest.main()
