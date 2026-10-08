"""Benchmark suite."""
from .registry import register_benchmark, list_benchmarks, run_all
from .metrics import recall_at_k, average_precision, mean_average_precision

# explicit imports so decorators run
from . import multimodal
from . import crossmodal
from . import matryoshka
from . import quant
from . import late

__all__ = [
    "register_benchmark", "list_benchmarks", "run_all",
    "recall_at_k", "average_precision", "mean_average_precision",
]
