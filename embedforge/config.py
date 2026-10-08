"""Configuration loading with a zero-dependency YAML fallback."""
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False


def _load_yaml(path):
    if _HAS_YAML:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    from .utils.yamlish import parse
    with open(path, "r", encoding="utf-8") as f:
        return parse(f.read()) or {}


@dataclass
class ModelConfig:
    dim: int = 64                 # production EmbeddingGemma 2 uses 768; runnable ref uses 64
    n_layers: int = 2
    n_heads: int = 4
    hidden: int = 128
    vocab: int = 512              # byte-level tokenizer alphabet
    patch: int = 4                # image patch size
    img_h: int = 8                # runnable image height
    img_w: int = 8                # runnable image width
    audio_frames: int = 16        # spectrogram frames
    mrl_dims: tuple = (8, 16, 32, 64)  # nested Matryoshka widths
    prod_dim_hint: str = "768"    # production embedding width (documented)


@dataclass
class BackendConfig:
    name: str = "mock"
    seed: int = 0
    base_url: str = ""
    api_key: str = ""
    model: str = ""


@dataclass
class TrainConfig:
    epochs: int = 3
    lr: float = 0.1
    steps: int = 30
    batch: int = 8
    temperature: float = 0.07


@dataclass
class RetrievalConfig:
    top_k: int = 10
    pq_subvectors: int = 8
    pq_centroids: int = 256


@dataclass
class ServeConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    auth_token: str = ""


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    serve: ServeConfig = field(default_factory=ServeConfig)

    @staticmethod
    def from_dict(d):
        m = d.get("model", {}) or {}
        b = d.get("backend", {}) or {}
        t = d.get("train", {}) or {}
        r = d.get("retrieval", {}) or {}
        s = d.get("serve", {}) or {}
        if "mrl_dims" in m and not isinstance(m["mrl_dims"], (list, tuple)):
            m["mrl_dims"] = [int(x) for x in str(m["mrl_dims"]).replace("[", "").replace("]", "").split(",") if x.strip()]
        return Config(
            model=ModelConfig(**{k: v for k, v in m.items() if k in ModelConfig.__dataclass_fields__}),
            backend=BackendConfig(**{k: v for k, v in b.items() if k in BackendConfig.__dataclass_fields__}),
            train=TrainConfig(**{k: v for k, v in t.items() if k in TrainConfig.__dataclass_fields__}),
            retrieval=RetrievalConfig(**{k: v for k, v in r.items() if k in RetrievalConfig.__dataclass_fields__}),
            serve=ServeConfig(**{k: v for k, v in s.items() if k in ServeConfig.__dataclass_fields__}),
        )


def load_config(path=None):
    if path is None:
        path = Path(__file__).resolve().parent / "configs" / "default.yaml"
    p = Path(path)
    data = {}
    if p.exists():
        data = _load_yaml(p)
    return Config.from_dict(data)
