# 配方：端侧低内存部署

## 目标

把嵌入模型压到移动端可用（~191MB 活跃内存）。

## 步骤

1. 用 Matryoshka 头取 64/128 维截断嵌入（省内存）。
2. 对权重做 INT4 group-wise 量化（`quant/int4.py`）。
3. 用 `quant/memory.py` 估算并验证内存目标。

```bash
python -m embedforge quant
python examples/06_quantization.py
```

## 关键点

- 归一化 + 量化是绝配，cosine 检索精度保持极高。
- 截断到低维后务必重新 L2 归一化。
