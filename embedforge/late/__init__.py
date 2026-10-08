"""Late interaction (ColBERT-style) retrieval."""
from .maxsim import MaxSim
from .colbert import ColBERTIndex

__all__ = ["MaxSim", "ColBERTIndex"]
