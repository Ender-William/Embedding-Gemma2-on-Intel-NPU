from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

# 直接加载本地 clone 下来的模型目录
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = "bfloat16" if device == "cuda" and torch.cuda.is_bf16_supported() else "float32"
model = SentenceTransformer(
    str(
        Path(__file__).resolve().parent.parent
        / "embedding_service"
        / "model"
        / "google--embeddinggemma-2"
    ),
    device=device,
    model_kwargs={"dtype": dtype},
)

# 检索场景：query 和 document 使用模型自带的 prompt 前缀效果最佳
query = "太阳系中最大的行星是哪一颗？"
documents = [
    "木星是太阳系中体积最大的行星。",
    "今天晚饭吃什么呢？",
    "Python 是一种流行的编程语言。",
]

query_emb = model.encode(query, prompt_name="query")
doc_embs = model.encode(documents, prompt_name="document")

# 向量维度 768，已做 L2 归一化
print("embedding shape:", doc_embs.shape)

scores = model.similarity(query_emb, doc_embs)[0]
for doc, score in zip(documents, scores):
    print(f"{score:.4f}  {doc}")
