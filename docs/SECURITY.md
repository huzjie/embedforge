# 安全说明

- 本仓库是 EmbeddingGemma 2 工程思想的独立参考实现，不含官方权重。
- 不要在 `config.yaml` 或代码里硬编码 API 密钥；用环境变量注入。
- HTTP 服务可设 `serve.auth_token` 启用 Bearer 鉴权。
- `openai` 后端通过 stdlib urllib 走 HTTPS，超时 60s。
