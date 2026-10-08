# 排障

| 症状 | 原因 | 解法 |
|---|---|---|
| `openai backend requires backend.api_key` | 未配置密钥 | 填 `config.yaml` 的 `backend.api_key` |
| `unknown backend 'x'` | 后端名拼写错误 | 用 `list_backends()` 查可用名 |
| Matryoshka 截断后相似度异常 | 截断后未重新归一化 | 用 `l2_norm(truncate(...))` |
| 检索 recall 恒 1.0 | 信号/噪声配比导致饱和 | 噪声务必满幅 [0,1]，见 mock 注释 |
| 图像编码报 `could not convert string to float` | 传入了字符串 | 传像素列表，或让 `_pixels` 走字符串分支 |
