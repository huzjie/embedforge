# 迟交互检索（Late Interaction / ColBERT）

## 问题

把整篇文档池化成**一个**向量，长文档的关键句会被淹没；于是传统方案要先做 chunking、OCR、版面切分。复杂且脆弱。

## 思路

**不池化，给每个 token 存一个向量**。打分时：

```
MaxSim(query_tokens, doc_tokens)
  = sum over query token q of  max over doc token d of  cosine(q, d)
```

一个查询词能命中文档的**任意位置**，因此文档无需预切块、无需 OCR。

## 可复用要点

1. **查询侧 token 少、文档侧 token 多**：MaxSim 的复杂度是 `O(nq * nd)`，查询短所以快。
2. **免 chunking 的核心收益**：让检索粒度从"段落"降到"token"，版面/表格里散落的关键信息也能被命中。
3. 代价是存储量从"每文档 1 个向量"变成"每文档 n 个向量"，需要 PQ 或倒排压缩（见 `docs/QUANTIZATION.md`）。

本项目实现见 `late/maxsim.py`、`late/colbert.py`，对比基准在 `bench/late.py`。
