"""Benchmark registry + run_all dispatcher."""
_BENCHMARKS = {}


def register_benchmark(name):
    def deco(fn):
        _BENCHMARKS[name] = fn
        return fn
    return deco


def list_benchmarks():
    return sorted(_BENCHMARKS)


def run_all(backend=None, cfg=None, **kwargs):
    """Run every registered benchmark, dispatching only the kwargs each needs."""
    results = {}
    for name, fn in _BENCHMARKS.items():
        results[name] = fn(backend=backend, cfg=cfg, **kwargs)
    return results
