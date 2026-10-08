# 常见问题

**Q: 为什么参考实现用 dim=64 而不是 768？**
A: 纯 Python 大维度矩阵乘法是 O(dim²) 循环，768 维会把 demo 拖慢到分钟级。生产 768 维的口径由成本模型（`quant/memory.py`）解析还原，见 `docs/SCALING.md`。

**Q: 如何切到真实嵌入模型？**
A: 把 `config.yaml` 的 `backend.name` 改为 `openai`/`vllm`/`transformers`，填 `base_url/api_key/model`。

**Q: mock 后端的 skill 是什么？**
A: 一个 [0,1] 标量，控制嵌入质量（真实语义 vs 噪声的比例）。训练让它单调趋 1，直接推动检索 recall 上升。

**Q: 迟交互和池化怎么选？**
A: 长文档、散落关键信息、不想 chunking → 迟交互；短文本、追求省存储 → 池化 + PQ。
