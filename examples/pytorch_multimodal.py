"""EmbeddingGemma 2 多模态编码演示（PyTorch 路径）。

验证 Text / Image 两种模态进入同一个 768 维向量空间，
并可跨模态计算相似度（文本 query 直接检索图片）。
"""

import math
import struct
import wave
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

ASSETS = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
ASSETS.mkdir(exist_ok=True)

# ---- 准备测试素材：纯色图片 + 正弦波音频 ----
Image.new("RGB", (640, 640), (220, 30, 30)).save(ASSETS / "red.png")
Image.new("RGB", (640, 640), (30, 30, 220)).save(ASSETS / "blue.png")

for name, freq in [("tone440.wav", 440.0), ("tone1000.wav", 1000.0)]:
    rate = 16000
    samples = [int(12000 * math.sin(2 * math.pi * freq * t / rate)) for t in range(rate * 2)]
    with wave.open(str(ASSETS / name), "wb") as w:
        w.setnchannels(1)  # 官方要求：单声道 16kHz
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack(f"<{len(samples)}h", *samples))

# ---- 加载模型（bf16，GPU 支持的話）----
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
model = SentenceTransformer(
    str(
        Path(__file__).resolve().parent.parent
        / "embedding_service"
        / "model"
        / "google--embeddinggemma-2"
    ),
    device="cuda" if torch.cuda.is_available() else "cpu",
    model_kwargs={"dtype": str(dtype).split(".")[-1]},
    config_kwargs={"audio_config": None},  # 只用文本+图像时可以不加载 audio tower，省显存
)

# ---- 1) 纯文本（带任务前缀）----
emb_text = model.encode("一张红色的方块图片", prompt_name="SearchQuery")

# ---- 2) 纯图像（图像不加任何前缀，用 <|image|> 占位）----
emb_imgs = model.encode(
    [
        {"text": "<|image|>", "image": [str(ASSETS / "red.png")]},
        {"text": "<|image|>", "image": [str(ASSETS / "blue.png")]},
    ]
)

# ---- 3) 图文交错（单条输入混合两种模态，输出一个向量）----
emb_mixed = model.encode(
    {"text": "这是什么颜色？<|image|>", "image": [str(ASSETS / "red.png")]}
)

print("text emb shape:", emb_text.shape)
print("image emb shape:", emb_imgs.shape)
print("mixed emb shape:", emb_mixed.shape)

# ---- 跨模态相似度：文本 query 检索图片 ----
sims = model.similarity(emb_text, emb_imgs)[0]
print(f"\nquery '一张红色的方块图片' vs 红图: {sims[0]:.4f}")
print(f"query '一张红色的方块图片' vs 蓝图: {sims[1]:.4f}")
print(f"query vs 图文混合输入(红图): {model.similarity(emb_text, emb_mixed)[0][0]:.4f}")

# ---- 4) MRL 截断：768 -> 128 维（截断后必须重新归一化）----
emb_128 = model.encode(
    "一张红色的方块图片", prompt_name="SearchQuery", truncate_dim=128, normalize_embeddings=True
)
print("\ntruncated emb shape:", emb_128.shape)
