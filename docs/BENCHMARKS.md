# 基准说明

五组基准，全部可复现、口径明确：

| 基准 | 指标 | 说明 |
|---|---|---|
| `multimodal_retrieval` | recall@1 / recall@5 / mAP | 文本查图像，recall 随 skill 0.5→1.0 |
| `crossmodal` | crossmodal_recall@1 | 文本查混合模态库 |
| `matryoshka` | ranking_agreement_by_width | 各宽度 top-k 排序与全维的一致性 |
| `quantization` | int8/int4 mse + cosine_retention | 量化精度保持 |
| `late_vs_pool` | 正样本排名 | 迟交互 vs 池化检索 |

运行：`python -m embedforge bench`。mock 后端用 `skill` 控制质量，训练后 recall@1 0.5→1.0。
