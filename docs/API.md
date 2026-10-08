# API 参考

## 命令行

```bash
python -m embedforge doctor          # 自检：后端/编码器/基准
python -m embedforge embed "文本" --modality text
python -m embedforge train           # 训练 mock，看 recall 0.5 -> 1.0
python -m embedforge bench           # 跑五组基准
python -m embedforge mrl             # 列出 Matryoshka 宽度
python -m embedforge quant           # 量化 + 内存报告
python -m embedforge search "查询"   # 向量检索
python -m embedforge serve --port 8000
```

## HTTP 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 |
| GET | `/v1/models` | 模型列表 |
| POST | `/v1/embeddings` | OpenAI 兼容嵌入 `{"input":..., "modality":...}` |
| POST | `/v1/matryoshka` | 按宽度嵌入 `{"input":..., "width":64}` |

## Python 接口

```python
from embedforge.config import load_config
from embedforge.model import UnifiedEmbedModel

cfg = load_config()
model = UnifiedEmbedModel(cfg.model)
emb = model.embed("text", "hello")          # 统一嵌入
emb_w = model.embed("text", "hello", width=32)  # Matryoshka 截断
```

## 后端注册

```python
from embedforge.backends import get_backend
b = get_backend("mock", cfg)   # mock / cpu / openai / vllm / transformers
b.embed("text", "hello")
```
