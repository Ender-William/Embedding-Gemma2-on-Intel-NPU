"""Minimal EmbeddingService example."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from embedding_service import EmbeddingService

service = EmbeddingService(
    models_root=ROOT / "embedding_service" / "model",
    cache_root=ROOT / "cache",
    default_device="cuda",
)

# The repository already contains exported artifacts under embedding_service/model/.
service.load(ROOT / "embedding_service" / "model" / "onnx", device="cuda")

queries = ["太阳系中最大的行星是哪一颗？"]
documents = [
    "木星是太阳系中体积最大的行星。",
    "今天晚饭吃什么好呢？",
    "Python 是一种流行的编程语言。",
]

query = service.run(queries, prompt="query")
docs = service.run(documents, prompt="document")
scores = service.similarity(docs.embeddings, query.embeddings)[:, 0]

print("embedding shape:", docs.embeddings.shape)
for index in scores.argsort()[::-1]:
    print(f"{scores[index]:.4f}  {documents[index]}")
