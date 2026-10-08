import unittest

from embedforge.config import load_config
from embedforge.backends.mock import MockBackend
from embedforge.train.trainer import MockTrainer
from embedforge.train.losses import contrastive_loss, siglip_loss


class TestTrain(unittest.TestCase):
    def test_skill_monotonic(self):
        cfg = load_config()
        mock = MockBackend(cfg)
        t = MockTrainer(mock, lr=0.1)
        start = mock.skill
        final = t.train(steps=20)
        self.assertGreater(final, start)
        self.assertLessEqual(final, 1.0)

    def test_losses_finite(self):
        a = [0.1] * 8
        b = [0.2] * 8
        self.assertGreater(contrastive_loss(a, b), 0)
        self.assertGreater(siglip_loss(a, b), 0)


if __name__ == "__main__":
    unittest.main()
