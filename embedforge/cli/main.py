"""Command-line interface for embedforge."""
import argparse
import json
import sys

from ..config import load_config
from ..backends import get_backend, list_backends
from ..utils.logging import get_logger

log = get_logger("embedforge.cli")


def _cmd_doctor(cfg, args):
    from ..utils.logging import get_logger as _gl
    _gl("embedforge.doctor")
    print(f"config: dim={cfg.model.dim} hidden={cfg.model.hidden} mrl={list(cfg.model.mrl_dims)}")
    print(f"backends available: {list_backends()}")
    for name in list_backends():
        try:
            b = get_backend(name, cfg)
            v = b.embed("text", "hello embedforge")
            print(f"  {name}: ok (dim={len(v)})")
        except Exception as e:
            print(f"  {name}: SKIP ({e})")
    # encoders + model
    from ..model import UnifiedEmbedModel
    m = UnifiedEmbedModel(cfg.model)
    for mod in m.modalities:
        v = m.embed(mod, "probe")
        print(f"  encoder[{mod}]: ok (dim={len(v)})")
    # benchmarks
    from ..bench import list_benchmarks
    print(f"benchmarks available: {list_benchmarks()}")
    print("doctor: OK")


def _cmd_embed(cfg, args):
    b = get_backend(cfg.backend.name, cfg)
    v = b.embed(args.modality, args.input)
    print(json.dumps({"modality": args.modality, "input": args.input,
                      "dim": len(v), "embedding": v[:8]}, ensure_ascii=False))


def _cmd_train(cfg, args):
    from ..backends.mock import MockBackend
    from ..train.trainer import MockTrainer
    from ..bench import run_all
    mock = MockBackend(cfg)
    print(f"skill before training: {mock.skill:.3f}")
    before = run_all(backend=mock, cfg=cfg)["multimodal_retrieval"]
    print(f"multimodal recall@1 before: {before['recall@1']}")
    t = MockTrainer(mock, lr=cfg.train.lr)
    final = t.train(steps=cfg.train.steps)
    print(f"skill after training: {final:.3f}")
    after = run_all(backend=mock, cfg=cfg)["multimodal_retrieval"]
    print(f"multimodal recall@1 after: {after['recall@1']}")
    print(f"recall@1: {before['recall@1']} -> {after['recall@1']}")


def _cmd_bench(cfg, args):
    b = get_backend(cfg.backend.name, cfg)
    from ..bench import run_all
    results = run_all(backend=b, cfg=cfg)
    for name, r in results.items():
        print(f"{name}: {json.dumps(r, ensure_ascii=False)}")


def _cmd_serve(cfg, args):
    from ..serving import serve
    serve(cfg, host=args.host, port=args.port)


def _cmd_index(cfg, args):
    from ..retrieval import VectorIndex
    b = get_backend(cfg.backend.name, cfg)
    idx = VectorIndex(cfg.model.dim)
    docs = args.docs or ["apple is a fruit", "banana is yellow", "car is a vehicle"]
    for i, d in enumerate(docs):
        idx.add(f"doc-{i}", b.embed("text", d), {"title": d})
    out = {f"doc-{i}": d for i, d in enumerate(docs)}
    print(f"indexed {len(idx)} documents")
    return idx, b, out


def _cmd_search(cfg, args):
    from ..retrieval import VectorIndex
    b = get_backend(cfg.backend.name, cfg)
    idx = VectorIndex(cfg.model.dim)
    docs = args.docs or ["apple is a fruit", "banana is yellow", "car is a vehicle"]
    for i, d in enumerate(docs):
        idx.add(f"doc-{i}", b.embed("text", d), {"title": d})
    q = b.embed("text", args.query)
    hits = idx.search(q, top_k=cfg.retrieval.top_k)
    for vid, score in hits:
        print(f"  {score:.4f}  {idx.meta[vid].get('title', vid)}")


def _cmd_mrl(cfg, args):
    from ..model import EmbedModelWithMRL
    m = EmbedModelWithMRL(cfg.model)
    print(f"mrl widths: {m.mrl_dims}")
    for w in m.mrl_dims:
        v = m.embed("text", "matryoshka demo", width=w)
        print(f"  width={w}: dim={len(v)}")


def _cmd_quant(cfg, args):
    from ..quant import MemoryTracker
    mt = MemoryTracker(cfg.model.dim, cfg.model.hidden, cfg.model.vocab, cfg.model.n_layers)
    print(mt.report())


def _build_parser():
    p = argparse.ArgumentParser(prog="embedforge", description="unified multimodal embedding + retrieval framework")
    p.add_argument("--config", default=None, help="path to config yaml")
    sub = p.add_subparsers(dest="cmd")

    sp = sub.add_parser("doctor")
    sp.set_defaults(func=_cmd_doctor)

    sp = sub.add_parser("embed")
    sp.add_argument("input")
    sp.add_argument("--modality", default="text", choices=["text", "image", "audio", "code"])
    sp.set_defaults(func=_cmd_embed)

    sp = sub.add_parser("train")
    sp.set_defaults(func=_cmd_train)

    sp = sub.add_parser("bench")
    sp.set_defaults(func=_cmd_bench)

    sp = sub.add_parser("serve")
    sp.add_argument("--host", default=None)
    sp.add_argument("--port", type=int, default=None)
    sp.set_defaults(func=_cmd_serve)

    sp = sub.add_parser("index")
    sp.add_argument("--docs", nargs="*", default=None)
    sp.set_defaults(func=_cmd_index)

    sp = sub.add_parser("search")
    sp.add_argument("query")
    sp.add_argument("--docs", nargs="*", default=None)
    sp.set_defaults(func=_cmd_search)

    sp = sub.add_parser("mrl")
    sp.set_defaults(func=_cmd_mrl)

    sp = sub.add_parser("quant")
    sp.set_defaults(func=_cmd_quant)

    return p


def main(argv=None):
    p = _build_parser()
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    if not hasattr(args, "func"):
        p.print_help()
        return 1
    args.func(cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
