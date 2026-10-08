"""Backend registry."""
_BACKENDS = {}


def register_backend(name):
    def deco(cls):
        _BACKENDS[name] = cls
        return cls
    return deco


def get_backend(name, cfg):
    if name not in _BACKENDS:
        raise ValueError(f"unknown backend '{name}'. available={sorted(_BACKENDS)}")
    return _BACKENDS[name](cfg)


def list_backends():
    return sorted(_BACKENDS)
