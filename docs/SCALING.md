# 规模与配置说明

## 运行态 vs 生产态

本仓库的**可运行参考**用 `dim=64 / hidden=128 / img 8x8`，保证纯 Python 在 CPU 上几秒跑完端到端。EmbeddingGemma 2 的**生产口径**是：

| 指标 | 生产值 | 参考值 |
|---|---|---|
| 嵌入维度 dim | 768 | 64 |
| 参数量 | ~740M（文本子模型 270M） | 极小 |
| 模态 | text/code/image/audio/video | text/code/image/audio |
| 许可 | Apache 2.0 | Apache 2.0 |
| 端侧内存 | ~191MB（INT4 量化） | 由 `quant/memory.py` 估算 |

## 为什么不在纯 Python 里跑 768 维

大维度矩阵乘法在纯 Python 里是 O(dim²) 的 Python 循环，会把 demo 拖慢到分钟级。**成本模型**（`quant/memory.py` 的 `footprint_bytes(prod=True)` 与 `bench/quant.py`）用解析方式还原生产数字，真实矩阵只在参考维度实例化。这是本项目家族反复验证的"规模靠成本模型、逻辑靠小模型"套路。

## 如何切到生产后端

把 `config.yaml` 的 `backend` 切成 `openai`/`vllm`/`transformers` 并填 `base_url/api_key/model`，即可在真实 768 维嵌入上跑同一套检索/量化/迟交互代码。
