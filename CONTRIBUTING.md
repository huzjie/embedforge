# 贡献指南

1. 保持零依赖：核心包只依赖 Python 标准库（可选依赖 `transformers` 仅在 transformers 后端使用）。
2. 新增模态 = 新增 `encoders/<modality>.py` + 在 `model/embed.py` 注册，下游自动可用。
3. 新增基准在 `bench/<name>.py` 用 `@register_benchmark` 注册，并在 `bench/__init__.py` 显式 import。
4. 提交前跑 `python -m compileall -q embedforge && python -m unittest discover -s tests -t .`。
