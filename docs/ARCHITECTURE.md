# 架构设计

## 总览

embedforge 把 EmbeddingGemma 2 背后的工程思想拆成一条可复用的流水线：

```
输入内容(text/code/image/audio)
   -> 模态编码器(encoders)          # 各自把内容转成 hidden 向量序列
   -> 共享投影头(projection)         # hidden -> 统一 dim -> L2 归一化
   -> 统一嵌入空间(一个向量空间)
   -> 检索/量化/迟交互 下游
```

核心一句话：**所有模态共享同一个投影头和同一个向量空间**，跨模态检索就是一次余弦相似度，不需要任何模态专用管道。

## 模块划分

| 模块 | 职责 | 关键文件 |
|---|---|---|
| `encoders` | 各模态编码（文本/图像/音频/代码） | `encoders/{text,image,audio,code}.py` |
| `projection` | 共享投影头 + Matryoshka 头 | `projection/{head,matryoshka}.py` |
| `model` | 统一嵌入模型 / 带 MRL 的模型 | `model/{embed,matryoshka_model}.py` |
| `train` | 对比损失(CLIP/SigLIP/MRL) + 训练器 | `train/{losses,dataset,trainer}.py` |
| `late` | 迟交互 MaxSim + ColBERT 索引 | `late/{maxsim,colbert}.py` |
| `retrieval` | 向量索引 / PQ / 跨模态 / RAG | `retrieval/*.py` |
| `quant` | INT8/INT4 量化 + 内存估算 | `quant/*.py` |
| `backends` | 五后端（mock/cpu/openai/vllm/transformers） | `backends/*.py` |
| `bench` | 五组基准 | `bench/*.py` |
| `serving` | stdlib HTTP 服务（OpenAI 兼容） | `serving/server.py` |

## 数据流（统一嵌入）

1. `UnifiedEmbedModel.embed(modality, content)` 按模态路由到编码器。
2. 编码器 `encode_tokens` 返回 `(n_tokens, hidden)` 的向量序列；`encode` 做 mean-pool 得到单个 `hidden` 向量。
3. `SharedProjectionHead.project` 用一层 `Linear(hidden -> dim)` 映射后 L2 归一化。

## 为什么"一个空间"是核心竞争力

- **跨模态检索零成本**：文搜图、图搜文、文搜音，全部是同一套余弦相似度。
- **可扩展**：新增一种模态 = 新增一个编码器 + 接入共享头，下游全部自动可用。
- **统一量化**：只量化一个投影头和共享层，而不是每种模态各一套。

## 可训练 Mock 的设计

mock 后端用一个标量 `skill` 控制嵌入质量：`emb = normalize(skill*gold + (1-skill)*noise)`。
- `gold` 只由内容（语义主题）决定，所以同主题跨模态正样本共享同一方向；
- 训练时 `skill` 单调趋 1（delta 恒正），检索 recall 从 ~0.5 爬到 1.0。

这个设计让"训练"这个动作真正影响下游指标，而不是装饰。
