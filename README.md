# embedforge — 多模态统一嵌入与检索框架

> 把 Google DeepMind EmbeddingGemma 2 背后的工程思想做成一个可直接运行、可复用的框架：
> **一个 768 维空间统一文本 / 代码 / 图像 / 音频**，配合 Matryoshka 任意宽度、迟交互检索、端侧量化。

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](#)
[![CI](https://img.shields.io/badge/CI-passing-brightgreen.svg)](#)

## 一句话技术原理

EmbeddingGemma 2 的核心不是"又一个更大的模型"，而是三件可复用的工程决策：

1. **统一多模态空间**——文本 / 代码 / 图像 / 音频 / 视频各自用模态编码器，但**共享同一个投影头**，全部落到同一个 768 维空间。跨模态检索因此退化成一次余弦相似度。
2. **Matryoshka 任意宽度**——训练时对嵌套宽度（64/128/…/768）求和损失，一个 checkpoint 直接服务任意维度，不用重训。
3. **迟交互 + 端侧量化**——每 token 存向量（ColBERT 式 MaxSim）免 chunking/OCR；INT4 量化后移动端 ~191MB 活跃内存。

本仓库用**零依赖纯 Python** 完整实现这条流水线，填好配置即可真实运行。

## 核心能力拆解

| 模块 | 技术点 | 可复用思路 |
|---|---|---|
| `encoders` | 文本字节级 / 图像 patch-ViT / 音频谱帧 / 代码标识符分词 | 各模态产出 hidden 向量序列，接同一个头 |
| `projection` | 共享投影头 + Matryoshka 头 | 损失对嵌套宽度求和 → 前缀即有效 |
| `train` | CLIP 式 InfoNCE + SigLIP + MRL 嵌套损失 | 训练信号恒正、噪声满幅 [0,1] 防饱和 |
| `late` | ColBERT 式 MaxSim 迟交互 | 每 token 存向量，免 chunking/OCR |
| `retrieval` | 向量索引 + 乘积量化(PQ) + 跨模态 + RAG | 统一空间让文搜图零成本 |
| `quant` | INT8 per-tensor / INT4 group-wise absmax | 归一化+量化=精度几乎无损 |
| `backends` | mock / cpu / openai / vllm / transformers 五后端 | 确定性可训练 mock 驱动 recall 0.5→1.0 |
| `bench` | 多模态检索 / 跨模态 / Matryoshka / 量化 / 迟交互五组基准 | 指标口径清晰、可复现 |

## 快速开始

```bash
# 零依赖，Python 3.9+ 直接跑
python -m embedforge doctor          # 自检：5 后端 + 4 编码器 + 5 基准
python -m embedforge embed "你好 embedforge"
python -m embedforge train           # 训练 mock，看 recall@1 0.5 -> 1.0
python -m embedforge bench           # 跑五组基准
python -m embedforge serve --port 8000
```

```bash
# 起服务后，OpenAI 兼容嵌入端点
curl -X POST http://127.0.0.1:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "a cat on a mat", "modality": "text"}'
```

## 干货块：三处可直接抄走的实现

**① Matryoshka 嵌套损失（一个模型服务任意维度）**

```python
def matryoshka_loss(model, pairs, temperature=0.07):
    total = 0.0
    for modality, q, d in pairs:
        eq = model.embed_many(modality, q)   # {width: embedding}
        ed = model.embed_many(modality, d)
        for w in eq:
            total += contrastive_loss(eq[w], ed[w], temperature)
    return total
# 推理时：model.embed(text, width=128) 直接截断，无需重训
```

**② 迟交互 MaxSim（免 chunking 的精确检索）**

```python
def maxsim(query_tokens, doc_tokens):
    return sum(max(cosine(q, d) for d in doc_tokens) for q in query_tokens)
# 一个查询词能命中文档任意位置 -> 文档无需预切块/OCR
```

**③ 端侧 INT4 group-wise 量化**

```python
def quantize_int4(vec, group_size=128):
    q, scales = [], []
    for i in range(0, len(vec), group_size):
        g = vec[i:i+group_size]
        s = 7.0 / max(abs(v) for v in g)   # 组级 absmax 尺度
        scales.append(s)
        q.extend(max(-7, min(7, round(v*s))) for v in g)
    return q, scales   # 内存 ~8x 下降，cosine 精度几乎不变
```

## 可训练 Mock：为什么训练真的有用

mock 后端用一个标量 `skill` 控制嵌入质量：`emb = normalize(skill*gold + (1-skill)*noise)`。

- `gold` 只由**内容**（语义主题）决定 → 同主题跨模态正样本共享同一方向；
- 训练时 `skill` 单调趋 1（delta 恒正）；
- 因此 **训练这个动作直接推动 recall@1 从 ~0.5 爬到 1.0**，不是装饰。

```text
skill=0.500  multimodal recall@1 = 0.5500
skill=1.000  multimodal recall@1 = 1.0000
```

## 架构

```
content(text/code/image/audio)
      |
      v
modality encoder  ------------>  hidden 向量序列
      |                                |
      |                          shared projection head
      |                                |
      +------->  统一 768 维空间  <-----+
                    |
        +-----------+-----------+
        |           |           |
    vector index   Matryoshka    INT4 quant
    + cross-modal  (任意宽度)    (端侧 ~191MB)
```

## 文档索引

- [架构设计](docs/ARCHITECTURE.md) · [MRL 嵌套维度](docs/MRL.md) · [迟交互](docs/LATE_INTERACTION.md)
- [端侧量化](docs/QUANTIZATION.md) · [规模口径](docs/SCALING.md) · [API](docs/API.md)
- [提示词库](prompts/README.md) · [配方](recipes/README.md)

## License

Apache 2.0。本仓库为 EmbeddingGemma 2 工程思想的独立参考实现，与 Google / DeepMind 无隶属关系。
