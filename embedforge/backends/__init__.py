"""Backend registry. Explicit imports so @register_backend decorators run."""
from .base import EmbeddingBackend
from .registry import register_backend, get_backend, list_backends

# explicit imports make the decorators execute
from . import mock
from . import cpu
from . import openai
from . import vllm
from . import transformers

__all__ = ["EmbeddingBackend", "register_backend", "get_backend", "list_backends"]
