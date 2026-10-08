import unittest

from embedforge.quant import quantize_int8, dequantize_int8, quantize_int4, dequantize_int4, MemoryTracker


class TestQuant(unittest.TestCase):
    def test_int8_roundtrip(self):
        v = [0.5, -0.3, 0.9, -0.1, 0.0, 1.0, -1.0, 0.2]
        q, s = quantize_int8(v)
        d = dequantize_int8(q, s)
        for a, b in zip(v, d):
            self.assertAlmostEqual(a, b, delta=0.02)

    def test_int4_roundtrip(self):
        v = [0.5, -0.3, 0.9, -0.1, 0.0, 1.0, -1.0, 0.2] * 2
        q, sc = quantize_int4(v, group_size=4)
        d = dequantize_int4(q, sc, group_size=4)
        for a, b in zip(v, d):
            self.assertAlmostEqual(a, b, delta=0.15)

    def test_memory(self):
        mt = MemoryTracker(64, 128, 512, 2)
        rep = mt.report()
        self.assertIn("MB", rep)


if __name__ == "__main__":
    unittest.main()
