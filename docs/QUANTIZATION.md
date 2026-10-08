# 端侧量化（INT8 / INT4）

## 思路

嵌入向量 L2 归一化后值域很小，适合低比特量化。两种粒度：

- **INT8（per-tensor absmax）**：`scale = 127 / max|x|`，`q = round(x*scale)`。最简单，几乎不损失余弦（归一化后余弦对尺度不敏感）。
- **INT4（group-wise absmax）**：把向量按 `group_size`（如 128）分组，每组单独算 scale。组级尺度能更好跟踪离群值，内存再降一半。

## 可复用要点

1. **量化粒度是精度/内存的杠杆**：per-tensor < per-group < per-channel，精度和内存成反比。
2. **归一化 + 量化是绝配**：嵌入先归一化再量化，cosine 检索精度保持极高。
3. 内存估算别用"每参数 32bit"，直接用 `params * bits / 8`。740M 参数：FP16 ~1.48GB、INT8 ~740MB、INT4 ~370MB；270M 文本子模型量化后移动端活跃内存 ~191MB（即官方口径）。

本项目实现见 `quant/{int8,int4,memory}.py`，基准在 `bench/quant.py`。
