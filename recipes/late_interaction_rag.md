# 配方：免 chunking 的迟交互 RAG

## 目标

长文档无需切块/OCR 就能被精确命中。

## 步骤

1. 用 `CpuBackend.embed_tokens` 为文档每个 token 存向量。
2. `ColBERTIndex.search(query_tokens)` 用 MaxSim 打分。

```bash
python examples/05_late_interaction.py
```
