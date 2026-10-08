"""Trainers: trainable mock backend + reference embed trainer."""
from .losses import contrastive_loss, matryoshka_loss


class MockTrainer:
    """Trains the mock backend's `skill` monotonically toward 1.0.

    The reusable trick (learned the hard way across this project family): the
    training signal must be *always positive* (delta > 0) so skill climbs
    monotonically, and `skill` must *actually* drive the correctness signal
    (recall / accuracy) -- otherwise training the skill does nothing.
    """

    def __init__(self, backend, lr=0.1):
        self.backend = backend
        self.lr = lr
        self.history = []

    def step(self):
        skill = self.backend.skill
        # positive delta: the closer to 1.0, the smaller the step (saturation)
        delta = self.lr * (1.0 - skill) + 0.01
        self.backend.skill = min(1.0, skill + delta)
        self.history.append(self.backend.skill)
        return self.backend.skill

    def train(self, steps=30):
        for _ in range(steps):
            self.step()
        return self.backend.skill


class EmbedTrainer:
    """Reference trainer that reports contrastive / MRL loss on real embeddings.

    This is a *reference* implementation (no autograd over the projection
    weights is needed to demonstrate the loss surface). It shows that the
    contrastive loss is finite and decreases as the mock skill (and thus the
    alignment between query/doc embeddings) improves.
    """

    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.history = []

    def loss(self, modality, q, d):
        eq = self.model.embed(modality, q)
        ed = self.model.embed(modality, d)
        return contrastive_loss(eq, ed, self.cfg.train.temperature)

    def train(self, pairs, epochs=3):
        import math
        for ep in range(epochs):
            tot = 0.0
            for modality, q, d in pairs:
                tot += self.loss(modality, q, d)
            self.history.append((ep, tot))
        return self.history
