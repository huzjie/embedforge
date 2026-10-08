import unittest

from embedforge.config import load_config, Config


class TestConfig(unittest.TestCase):
    def test_default(self):
        cfg = load_config()
        self.assertIsInstance(cfg, Config)
        self.assertGreater(cfg.model.dim, 0)
        self.assertIn(cfg.model.dim, cfg.model.mrl_dims)

    def test_mrl_nested(self):
        cfg = load_config()
        self.assertEqual(sorted(cfg.model.mrl_dims), cfg.model.mrl_dims)


if __name__ == "__main__":
    unittest.main()
