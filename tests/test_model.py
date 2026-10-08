import unittest

from embedforge.config import load_config
from embedforge.model import UnifiedEmbedModel, EmbedModelWithMRL


class TestModel(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config().model

    def test_unified_embed(self):
        m = UnifiedEmbedModel(self.cfg)
        for mod in m.modalities:
            v = m.embed(mod, "probe")
            self.assertEqual(len(v), self.cfg.dim)

    def test_mrl(self):
        m = EmbedModelWithMRL(self.cfg)
        v = m.embed("text", "probe")
        self.assertEqual(len(v), self.cfg.dim)
        for w in m.mrl_dims:
            vw = m.embed("text", "probe", width=w)
            self.assertEqual(len(vw), w)


if __name__ == "__main__":
    unittest.main()
