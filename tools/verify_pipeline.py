"""End-to-end verification of ONNX/OpenVINO multimodal pipelines.

Covers text, image, audio, video, mixed input, prompts, and long inputs.
"""

from __future__ import annotations

import math
import struct
import sys
import wave
from pathlib import Path

import av
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embedding_service.core.engine import EmbeddingEngine  # noqa: E402
from embedding_service.core.multimodal import (  # noqa: E402
    OnnxMultimodalEncoder,
    OpenVinoMultimodalEncoder,
    PROMPTS,
    decode_video_frames,
)

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "tests" / "fixtures"
ASSETS.mkdir(exist_ok=True)


def make_assets():
    rate = 16000
    samples = []
    for t in range(rate * 25):
        freq = 440.0 if (t // rate) % 2 == 0 else 880.0
        samples.append(int(10000 * math.sin(2 * math.pi * freq * t / rate)))
    with wave.open(str(ASSETS / "long25s.wav"), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack(f"<{len(samples)}h", *samples))

    path = ASSETS / "red_blue_3s.mp4"
    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=5)
    stream.width = 320
    stream.height = 240
    stream.pix_fmt = "yuv420p"
    for index in range(15):
        array = np.empty((240, 320, 3), dtype=np.uint8)
        array[:] = [220, 30, 30] if index < 7 else [30, 30, 220]
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def cosine(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


make_assets()

ref_model = SentenceTransformer(
    str(ROOT / "embedding_service" / "model" / "google--embeddinggemma-2"),
    device="cuda" if torch.cuda.is_available() else "cpu",
    model_kwargs={"dtype": "float32"},
)

ov_device = {"vision": "CPU", "audio": "NPU", "lm": "NPU"}
try:
    openvino_engine = OpenVinoMultimodalEncoder(
        ROOT / "embedding_service" / "model" / "openvino", device=ov_device, seq_len=512
    )
except Exception as exc:
    print("OpenVINO NPU unavailable, using CPU:", exc)
    openvino_engine = OpenVinoMultimodalEncoder(
        ROOT / "embedding_service" / "model" / "openvino", device="CPU", seq_len=1024
    )

engines = {
    "onnx": OnnxMultimodalEncoder(
        ROOT / "embedding_service" / "model" / "onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    ),
    "openvino": openvino_engine,
}

video_path = str(ASSETS / "red_blue_3s.mp4")
video_frames = decode_video_frames(video_path)

CASES = [
    ("text+prompt", "一张红色的方块图片", None, "query"),
    ("image-red", {"text": "<|image|>", "image": [str(ASSETS / "red.png")]}, None, None),
    ("image-wide", {"text": "<|image|>", "image": [str(ASSETS / "wide.png")]}, None, None),
    ("audio-2s", {"text": "<|audio|>", "audio": [str(ASSETS / "tone440.wav")]}, None, None),
    (
        "mixed+prompt",
        {"text": "这是什么？<|image|>", "image": [str(ASSETS / "red.png")]},
        None,
        "query",
    ),
    (
        "audio-25s",
        {"text": "<|audio|>", "audio": [str(ASSETS / "long25s.wav")]},
        None,
        None,
    ),
]

print(f"{'case':<20} {'onnx':>12} {'openvino':>12}")
for name, inp, ref_inp, prompt in CASES:
    if ref_inp is not None:
        reference_input = ref_inp
    elif prompt and isinstance(inp, str):
        reference_input = PROMPTS[prompt] + inp
    elif prompt and isinstance(inp, dict):
        reference_input = dict(inp)
        reference_input["text"] = PROMPTS[prompt] + reference_input["text"]
    else:
        reference_input = inp
    ref = ref_model.encode(reference_input)
    row = f"{name:<20}"
    for engine in engines.values():
        out = engine.encode(inp, prompt=prompt)
        row += f"{cosine(ref, out):>12.6f}"
    print(row)

# Video: PyTorch receives decoded PIL frames; pipeline receives the MP4 path.
video_inp = {"text": "红色到蓝色的变化 <|video|>", "video": video_path}
video_ref_inp = {"text": "红色到蓝色的变化 <|video|>", "video": video_frames}
video_ref = ref_model.encode(video_ref_inp)
print(f"{'video-path':<20}", end="")
for engine in engines.values():
    print(f"{cosine(video_ref, engine.encode(video_inp)):>12.6f}", end="")
print()

# Prompt must affect mixed text+media inputs.
image_inp = {"text": "这是什么颜色？<|image|>", "image": [str(ASSETS / "red.png")]}
a = engines["onnx"].encode(image_inp, prompt="query")
b = engines["onnx"].encode(image_inp, prompt="document")
assert np.linalg.norm(a - b) > 0.01, "prompt was ignored for multimodal input"

# Placeholder/media mismatch must be explicit and actionable.
try:
    engines["onnx"].encode({"text": "<|image|><|image|>", "image": [str(ASSETS / "red.png")]})
except ValueError as exc:
    print("placeholder validation:", exc)
else:
    raise AssertionError("placeholder mismatch was not rejected")

# Long text: ONNX must use the full 8192-token context, not 512.
long_text = "这是一段用于测试长文档上下文的资源库内容，包含产品说明、使用限制、兼容性和售后政策。" * 120
long_ref = ref_model.encode(long_text, prompt_name="SearchQuery")
long_engine = EmbeddingEngine(verbose=False)
long_out = long_engine.encode([long_text], prompt="query")[0]
print("long-text cosine:", f"{cosine(long_ref, long_out):.6f}")
assert cosine(long_ref, long_out) > 0.995

ov_engine = EmbeddingEngine(prefer=["openvino"], verbose=False)
ov_long_out = ov_engine.encode([long_text], prompt="query")[0]
print("openvino long-text cosine:", f"{cosine(long_ref, ov_long_out):.6f}")
assert cosine(long_ref, ov_long_out) > 0.995

print("ALL PIPELINE CHECKS PASSED")
