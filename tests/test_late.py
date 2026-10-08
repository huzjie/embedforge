import unittest

from embedforge.late import MaxSim, ColBERTIndex


class TestLate(unittest.TestCase):
    def test_maxsim_self_high(self):
        toks = [[0.5] * 4, [0.6] * 4]
        self.assertGreater(MaxSim.score(toks, toks), 0)

    def test_colbert(self):
        idx = ColBERTIndex()
        idx.add("a", [[0.1] * 4])
        idx.add("b", [[0.2] * 4])
        hits = idx.search([[0.1] * 4], top_k=2)
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0][0], "a")


if __name__ == "__main__":
    unittest.main()
