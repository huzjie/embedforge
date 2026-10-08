import unittest

from embedforge.config import load_config
from embedforge.backends import list_backends, get_backend


class TestBackends(unittest.TestCase):
    def test_registry(self):
        names = list_backends()
        self.assertIn("mock", names)
        self.assertIn("cpu", names)
        self.assertIn("openai", names)

    def test_mock_embed(self):
        cfg = load_config()
        b = get_backend("mock", cfg)
        v = b.embed("text", "hello")
        self.assertEqual(len(v), cfg.model.dim)

    def test_cpu_embed(self):
        cfg = load_config()
        b = get_backend("cpu", cfg)
        v = b.embed("text", "hello")
        self.assertEqual(len(v), cfg.model.dim)


if __name__ == "__main__":
    unittest.main()
