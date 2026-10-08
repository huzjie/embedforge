# 配方：跨模态检索

## 目标

用文本查询检索图片/音频库，零模态专用管道。

## 步骤

1. 用 `UnifiedEmbedModel` 把图/音文档索引进 `CrossModalSearch`。
2. 文本查询走 `search("text", query)`。
3. 命中结果带模态标签，直接呈现。

```bash
python examples/03_cross_modal_search.py
```

## 关键点

- 所有模态共享一个空间，查询和文档模态可以不同。
- 命中分数就是余弦相似度，天然可排序、可设阈值。
