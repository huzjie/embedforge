"""Training: contrastive losses, datasets, trainers."""
from .losses import contrastive_loss, siglip_loss, matryoshka_loss
from .dataset import MultiModalPairs
from .trainer import MockTrainer, EmbedTrainer

__all__ = [
    "contrastive_loss", "siglip_loss", "matryoshka_loss",
    "MultiModalPairs", "MockTrainer", "EmbedTrainer",
]
