"""Hit the stdlib HTTP server with an OpenAI-compatible embeddings call."""
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    url = "http://127.0.0.1:8000/v1/embeddings"
    payload = json.dumps({"input": "hello embedforge", "modality": "text"}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read().decode("utf-8"))
        emb = data["data"][0]["embedding"]
        print(f"embedding dim={len(emb)} head={[round(x, 4) for x in emb[:4]]}")
    except Exception as e:
        print(f"(server not running) {e}")


if __name__ == "__main__":
    main()
