import unittest

from embedforge.config import load_config
from embedforge.backends.cpu import CpuBackend
from embedforge.retrieval import CrossModalSearch, RAGPipeline


class TestRAG(unittest.TestCase):
    def test_answer(self):
        cfg = load_config()
        cpu = CpuBackend(cfg)
        index = CrossModalSearch(cpu)
        index.index("text", "doc about topic 1", "d1", {"title": "d1"})
        rag = RAGPipeline(cpu, index=index, generator=RAGPipeline.mock_generator)
        ans = rag.answer("topic 1", top_k=3)
        self.assertIn("d1", ans)


if __name__ == "__main__":
    unittest.main()
