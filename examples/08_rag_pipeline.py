"""Minimal RAG: retrieve + grounded answer (mock generator)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embedforge.config import load_config
from embedforge.backends.cpu import CpuBackend
from embedforge.retrieval import CrossModalSearch, RAGPipeline


def main():
    cfg = load_config()
    cpu = CpuBackend(cfg)
    index = CrossModalSearch(cpu)
    for i in range(5):
        index.index("text", f"document {i} explains concept {i} in detail", f"doc-{i}",
                    {"title": f"doc {i}"})
    rag = RAGPipeline(cpu, index=index, generator=RAGPipeline.mock_generator)
    print(rag.answer("concept 2"))


if __name__ == "__main__":
    main()
