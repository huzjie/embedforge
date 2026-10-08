"""Zero-dependency stdlib HTTP server exposing embeddings + retrieval.

Endpoints:
- GET  /health            -> {"status":"ok"}
- GET  /v1/models         -> model list
- POST /v1/embeddings     -> OpenAI-compatible {"data":[{"embedding":...}]}
- POST /v1/retrieval      -> {"action":"index"|"search", ...}
- POST /v1/matryoshka     -> embed at a nested width
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..backends import get_backend
from ..utils.logging import get_logger

log = get_logger("embedforge.serving")


class EmbeddingHTTPServer:
    def __init__(self, cfg, backend=None):
        self.cfg = cfg
        self.backend = backend or get_backend(cfg.backend.name, cfg)
        self.handler = self._make_handler()

    def _make_handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, obj):
                body = json.dumps(obj).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _read(self):
                n = int(self.headers.get("Content-Length", 0) or 0)
                return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}

            def _auth(self):
                tok = server.cfg.serve.auth_token
                if tok:
                    auth = self.headers.get("Authorization", "")
                    if auth != f"Bearer {tok}":
                        self._send(401, {"error": "unauthorized"})
                        return False
                return True

            def do_GET(self):
                if self.path == "/health":
                    self._send(200, {"status": "ok"})
                elif self.path == "/v1/models":
                    self._send(200, {"data": [{"id": server.backend.name, "object": "model"}]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                if not self._auth():
                    return
                if self.path == "/v1/embeddings":
                    data = self._read()
                    modality = data.get("modality", "text")
                    content = data.get("input", "")
                    emb = server.backend.embed(modality, content)
                    self._send(200, {
                        "object": "list",
                        "data": [{"object": "embedding", "embedding": emb, "index": 0}],
                        "model": server.backend.name,
                        "usage": {"prompt_tokens": len(str(content)), "total_tokens": len(str(content))},
                    })
                elif self.path == "/v1/matryoshka":
                    data = self._read()
                    modality = data.get("modality", "text")
                    content = data.get("input", "")
                    width = data.get("width")
                    if hasattr(server.backend, "model"):
                        emb = server.backend.model.embed(modality, content, width=width)
                    else:
                        emb = server.backend.embed(modality, content)
                        if width:
                            emb = emb[:width]
                    self._send(200, {"embedding": emb, "width": width})
                else:
                    self._send(404, {"error": "not found"})

        return Handler

    def run(self, host=None, port=None):
        host = host or self.cfg.serve.host
        port = port or self.cfg.serve.port
        httpd = ThreadingHTTPServer((host, port), self.handler)
        log.info(f"embedforge serving on http://{host}:{port}")
        return httpd


def serve(cfg, host=None, port=None):
    srv = EmbeddingHTTPServer(cfg)
    httpd = srv.run(host, port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
