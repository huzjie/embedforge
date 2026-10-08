# 训练数据清洗模板

```
判断这对 (query, document) 是否是高质量的正样本：
要求语义一致、非重复、非标题党。输出 JSON：{"positive": true/false, "reason": "..."}
query：{q}
document：{d}
```
