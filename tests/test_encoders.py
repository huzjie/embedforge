import unittest

from embedforge.config import load_config
from embedforge.encoders import TextEncoder, ImageEncoder, AudioEncoder, CodeEncoder


class TestEncoders(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config().model

    def test_text(self):
        e = TextEncoder(self.cfg)
        seq = e.encode_tokens("hello world")
        self.assertTrue(seq)
        self.assertEqual(len(seq[0]), self.cfg.hidden)

    def test_image(self):
        e = ImageEncoder(self.cfg)
        img = [[0.1] * self.cfg.img_w for _ in range(self.cfg.img_h)]
        seq = e.encode_tokens(img)
        self.assertTrue(seq)
        self.assertEqual(len(seq[0]), self.cfg.hidden)

    def test_audio(self):
        e = AudioEncoder(self.cfg)
        seq = e.encode_tokens([0.5] * 64)
        self.assertTrue(seq)
        self.assertEqual(len(seq[0]), self.cfg.hidden)

    def test_code(self):
        e = CodeEncoder(self.cfg)
        seq = e.encode_tokens("def f(x):\n    return x")
        self.assertTrue(seq)
        self.assertEqual(len(seq[0]), self.cfg.hidden)


if __name__ == "__main__":
    unittest.main()
