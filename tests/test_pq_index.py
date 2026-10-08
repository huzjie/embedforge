import unittest

from embedforge.retrieval import PQIndex, ProductQuantizer


class TestPQIndex(unittest.TestCase):
    def test_search(self):
        idx = PQIndex(16, m=4, k=16)
        idx.add("a", [1.0] * 16)
        idx.add("b", [-1.0] * 16)
        hits = idx.search([1.0] * 16, top_k=2)
        self.assertEqual(hits[0][0], "a")
        self.assertEqual(len(hits), 2)

    def test_bytes(self):
        pq = ProductQuantizer(64, m=8, k=256)
        self.assertLess(pq.bytes_per_vector(), 64 * 4)


if __name__ == "__main__":
    unittest.main()
