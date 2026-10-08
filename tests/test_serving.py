import json
import threading
import unittest
import urllib.request

from embedforge.config import load_config
from embedforge.serving import EmbeddingHTTPServer


class TestServing(unittest.TestCase):
    def test_health_and_embeddings(self):
        cfg = load_config()
        cfg.serve.port = 8099
        srv = EmbeddingHTTPServer(cfg)
        httpd = srv.run(host="127.0.0.1", port=8099)
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            with urllib.request.urlopen("http://127.0.0.1:8099/health", timeout=5) as r:
                self.assertEqual(json.loads(r.read())["status"], "ok")
            payload = json.dumps({"input": "hi", "modality": "text"}).encode("utf-8")
            req = urllib.request.Request("http://127.0.0.1:8099/v1/embeddings",
                                         data=payload, method="POST")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.loads(r.read())
            self.assertEqual(len(data["data"][0]["embedding"]), cfg.model.dim)
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
