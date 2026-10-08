# 部署

## Docker

```bash
docker build -f deploy/Dockerfile -t embedforge .
docker run -p 8000:8000 embedforge
```

## Kubernetes

```bash
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/service.yaml
```

## Helm

```bash
helm install embedforge deploy/helm
```

## 环境变量

- `EMBEDFORGE_BACKEND`: mock / cpu / openai / vllm / transformers
- `EMBEDFORGE_API_KEY`: openai/vllm 后端的密钥
- `EMBEDFORGE_BASE_URL`: 兼容端点地址
