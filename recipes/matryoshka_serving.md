# 配方：单模型多宽度服务

## 目标

一个 checkpoint 同时服务 64/128/256/768 维，不同租户不同宽度。

## 步骤

1. `EmbedModelWithMRL.embed(content, width=w)` 按需截断。
2. HTTP 端点 `/v1/matryoshka` 传 `width` 参数。

```bash
python -m embedforge mrl
python examples/04_matryoshka.py
```
