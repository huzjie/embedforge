"""Index an image/audio gallery, then search it with a TEXT query."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embedforge.config import load_config
from embedforge.retrieval import CrossModalSearch
from embedforge.model import UnifiedEmbedModel


def main():
    cfg = load_config()
    model = UnifiedEmbedModel(cfg.model)
    search = CrossModalSearch(model)
    for i in range(6):
        search.index("image", f"topic-{i}", f"img-{i}", {"title": f"image of topic {i}"})
    for i in range(3):
        search.index("audio", f"topic-{i}", f"aud-{i}", {"title": f"audio of topic {i}"})
    hits = search.search("text", "topic-3", top_k=5)
    for vid, score, modality, meta in hits:
        print(f"  {score:.4f}  {vid} ({modality})  {meta['title']}")


if __name__ == "__main__":
    main()
