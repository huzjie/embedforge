# 配方：乘积量化索引

## 目标

在内存受限下做大库检索。

## 步骤

1. 用 `ProductQuantizer` 把 d 维向量切成 m 个子向量。
2. 每个子向量量化到最近 centroid（k=256 → 每向量 ~m 字节）。
3. 检索用近似距离 + top-k 重排。

```bash
python examples/07_pq_index.py
```
