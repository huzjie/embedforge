# 配方：评测流水线

## 目标

可复现地评估检索质量。

## 步骤

1. 构造带相关标注的 query-doc 集合。
2. 跑 `python -m embedforge bench` 得 recall@k / mAP。
3. 用 Matryoshka 基准看宽度对排序一致性的影响。
4. 用量化基准看压缩对精度的影响。
