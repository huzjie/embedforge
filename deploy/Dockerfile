# embedforge - unified multimodal embedding + retrieval framework
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["python", "-m", "embedforge", "serve", "--host", "0.0.0.0", "--port", "8000"]
