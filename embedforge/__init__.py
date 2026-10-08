"""embedforge: a unified multimodal embedding + retrieval framework.

A zero-dependency, fully runnable reference implementation of the engineering
ideas behind Google DeepMind's EmbeddingGemma 2 (announced 2026-10-08): a single
~740M-parameter model that maps text, code, images, audio and video into ONE
768-dimensional space, runs on-device (~191 MB quantized), and is trained with
Matryoshka Representation Learning so a single checkpoint serves any embedding
width (64 / 128 / ... / 768).

This repo distills its most reusable engineering ideas into one framework:

1. Unified multimodal embedding -- modality-specific encoders (text tokenizer +
   transformer, image patch/ViT encoder, audio spectrogram encoder, code
   tokenizer) feeding ONE shared L2-normalized projection head.
2. Matryoshka Representation Learning (MRL) -- a single model trained with a
   nested sub-dimension loss is truncatable to any width without retraining.
3. Contrastive training -- in-batch CLIP-style softmax + SigLIP sigmoid losses
   with hard-negative mining.
4. Late interaction retrieval -- per-token vectors + MaxSim (ColBERT-style)
   scoring, so documents need no OCR / chunking.
5. On-device quantization -- INT8 / INT4 group-wise absmax with a memory tracker.
6. Multimodal retrieval / RAG -- vector index + product quantization + cross-
   modal search (text<->image<->audio).

Highlights
----------
- Zero-dependency pure-Python tensor core with autodiff over the op set
- Deterministic trainable mock backend (skill monotonically -> 1.0)
- Five benchmarks (multimodal / cross-modal / Matryoshka / quant / late-vs-pool)
- OpenAI-compatible stdlib HTTP serving + CLI + Docker/K8s/Helm/CI
- Prompt library + recipes + extensive docs
"""
from .version import __version__
from .config import Config, load_config

__all__ = ["Config", "load_config", "__version__"]
