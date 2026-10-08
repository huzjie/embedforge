import unittest

from embedforge.retrieval import VectorIndex, ProductQuantizer, PQIndex, CrossModalSearch
from embedforge.config import load_config
from embedforge.model import UnifiedEmbedModel


class TestRetrieval(unittest.TestCase):
    def test_vector_index(self):
        idx = VectorIndex(4)
        idx.add("a", [1.0, 0.0, 0.0, 0.0])
        idx.add("b", [0.0, 1.0, 0.0, 0.0])
        hits = idx.search([1.0, 0.0, 0.0, 0.0], top_k=1)
        self.assertEqual(hits[0][0], "a")

    def test_pq(self):
        pq = ProductQuantizer(16, m=4, k=16)
        v = [0.1] * 16
        codes = pq.encode(v)
        self.assertEqual(len(codes), 4)
        dec = pq.decode(codes)
        self.assertEqual(len(dec), 16)

    def test_cross_modal(self):
        cfg = load_config().model
        model = UnifiedEmbedModel(cfg)
        cs = CrossModalSearch(model)
        cs.index("image", "topic-0", "img0", {"title": "t0"})
        hits = cs.search("text", "topic-0", top_k=1)
        self.assertEqual(hits[0][0], "img0")


if __name__ == "__main__":
    unittest.main()
