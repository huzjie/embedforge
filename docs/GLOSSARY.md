# 术语

- **MRL (Matryoshka Representation Learning)**：嵌套维度学习，一个模型服务任意宽度。
- **Late Interaction**：迟交互检索，每 token 存向量，用 MaxSim 打分（ColBERT 式）。
- **PQ (Product Quantization)**：乘积量化，把向量切成子向量分别量化以省内存。
- **MaxSim**：迟交互打分 = 查询 token 对文档 token 最大余弦之和。
- **InfoNCE / CLIP loss**：in-batch 对比损失。
- **SigLIP**：用 sigmoid 而非 softmax 的对比损失，pairwise 计算。
