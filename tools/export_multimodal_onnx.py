"""导出多模态 ONNX 模型组：

    embedding_service/model/onnx/
      vision_encoder.onnx   图像: (pixel_values[1,P,768], image_position_ids[1,P,2]) -> [S,512] 软 token
      audio_encoder.onnx    音频: (input_features[1,T,128], input_features_mask[1,T]) -> [1,S,512] + 有效掩码
      lm_embeds.onnx        语言模型: (inputs_embeds[1,L,512], attention_mask[1,L]) -> [1,L,768]
      token_embeddings.npy  文本 token 嵌入表（bf16），host 侧查表后注入软 token

每个导出都用 onnxruntime 与 PyTorch 原模型做数值校验。
"""

import argparse
import os
import sys
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from embedding_service.core import runtime_env  # noqa: E402,F401
import numpy as np
import onnxruntime as ort
import torch
from ml_dtypes import bfloat16
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor

parser = argparse.ArgumentParser(description="Export EmbeddingGemma 2 multimodal ONNX artifacts")
parser.add_argument(
    "--model-dir",
    type=Path,
    default=ROOT / "embedding_service" / "model" / "google--embeddinggemma-2",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=ROOT / "embedding_service" / "model" / "onnx",
)
parser.add_argument(
    "--skip-vision",
    action="store_true",
    help="skip vision exports while debugging text/audio components",
)
args = parser.parse_args()

MODEL_DIR = args.model_dir.expanduser().resolve()
OUT_DIR = args.output_dir.expanduser().resolve()
if not MODEL_DIR.exists():
    raise SystemExit(f"model directory does not exist: {MODEL_DIR}")
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"model: {MODEL_DIR}")
print(f"output: {OUT_DIR}")

ASSETS = ROOT / "tests" / "fixtures"

# 准备两张不同长宽比的测试图，验证动态形状
ASSETS.mkdir(exist_ok=True)
Image.new("RGB", (640, 640), (220, 30, 30)).save(ASSETS / "red.png")
Image.new("RGB", (1280, 480), (30, 220, 30)).save(ASSETS / "wide.png")

# 注意：不能用 attn_implementation="eager"——音频塔的 eager 掩码路径有 bug
# （float 掩码 vs masked_fill(bool.logical_not()) 语义相反），sdpa 才是正确路径
st = SentenceTransformer(str(MODEL_DIR), device="cpu", model_kwargs={"dtype": "float32"})
backbone = st[0].auto_model
processor = AutoProcessor.from_pretrained(str(MODEL_DIR))


def preproc_image(path):
    msgs = [[{"role": "user", "content": [
        {"type": "text", "text": "这是什么？"},
        {"type": "image", "image": str(path)},
    ]}]]
    return processor.apply_chat_template(
        msgs, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=False
    )


def preproc_audio(path):
    msgs = [[{"role": "user", "content": [{"type": "audio", "audio": str(path)}]}]]
    return processor.apply_chat_template(
        msgs, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=False
    )


def export(module, args, path, input_names, output_names, dynamic_shapes, dynamo=True):
    if dynamo:
        torch.onnx.export(
            module, args, str(path),
            input_names=input_names, output_names=output_names,
            dynamic_shapes=dynamic_shapes, opset_version=17, dynamo=True,
        )
    else:
        # 旧版 TorchScript 导出器：dynamic_axes 需要连输出一起声明
        torch.onnx.export(
            module, args, str(path),
            input_names=input_names, output_names=output_names,
            dynamic_axes=dynamic_shapes, opset_version=17, dynamo=False,
        )
    print("exported:", path.name)


def ort_run(path, feeds):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return sess.run(None, feeds)


def close(a, b, tol=2e-3):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    if a.shape != b.shape:
        return False, f"shape {a.shape} vs {b.shape}"
    return np.abs(a - b).max() < tol, f"max diff {np.abs(a - b).max():.2e}"


SKIP_VISION = args.skip_vision or os.environ.get("SKIP_VISION") == "1"


# ============ 1) 视觉编码器 ============
class VisionEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_tower = backbone.vision_tower
        self.embed_vision = backbone.embed_vision

    def forward(self, pixel_values, image_position_ids):
        out = self.vision_tower(pixel_values=pixel_values, pixel_position_ids=image_position_ids)
        return self.embed_vision(out.last_hidden_state)  # [S, 512]（无 batch 维）


if not SKIP_VISION:
    vision = VisionEncoder().eval()
    f_img = preproc_image(ASSETS / "red.png")
    with torch.no_grad():
        ref_vis = vision(f_img["pixel_values"], f_img["image_position_ids"]).numpy()
        n_placeholder = int((f_img["input_ids"] == backbone.config.image_token_id).sum())
    print(f"vision soft tokens: {ref_vis.shape}, placeholders: {n_placeholder}")
    assert ref_vis.shape[0] == n_placeholder, "软 token 数与占位符数不一致"

    export(
        vision,
        (f_img["pixel_values"], f_img["image_position_ids"]),
        OUT_DIR / "vision_encoder.onnx",
        ["pixel_values", "image_position_ids"],
        ["image_embeds"],
        {
            "pixel_values": {1: "num_patches"},
            "image_position_ids": {1: "num_patches"},
        },
    )

    # 校验：方图 + 宽图（不同 patch 数）
    for name in ["red.png", "wide.png"]:
        f = preproc_image(ASSETS / name)
        with torch.no_grad():
            ref = vision(f["pixel_values"], f["image_position_ids"]).numpy()
        (out,) = ort_run(
            OUT_DIR / "vision_encoder.onnx",
            {"pixel_values": f["pixel_values"].numpy(), "image_position_ids": f["image_position_ids"].numpy()},
        )
        ok, msg = close(out, ref)
        print(f"  vision[{name}]: {msg} {'OK' if ok else 'FAIL'}")
        assert ok
else:
    print("vision export skipped (SKIP_VISION=1)")

# ============ 1.5) 视觉编码器（固定输出变体，供 OpenVINO/NPU 静态编译用）============
# 原版图在 pooler 后按 pooler_mask 裁剪掉 pad 行，输出形状随图片变化；
# 这个变体保留全部 280 行（pad 行也在），host 侧取前 S 行（S = 非pad patch数 / 9，已知前置连续）。
VISION_CANVAS = 2520  # max_soft_tokens(280) * pooling_kernel^2(9)，处理器画布上限
VISION_OUT = 280


class VisionEncoderFixed(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tower = backbone.vision_tower
        self.embed_vision = backbone.embed_vision

    def forward(self, pixel_values, image_position_ids):
        t = self.tower
        padding_positions = (image_position_ids == -1).all(dim=-1)
        emb = t.patch_embedder(pixel_values, image_position_ids, padding_positions)
        enc = t.encoder(
            inputs_embeds=emb,
            attention_mask=~padding_positions,
            pixel_position_ids=image_position_ids,
        )
        hidden, _ = t.pooler(
            hidden_states=enc.last_hidden_state,
            pixel_position_ids=image_position_ids,
            padding_positions=padding_positions,
            output_length=pixel_values.shape[-2] // 9,
        )
        return self.embed_vision(hidden)[0]  # [280, 512]，保留 pad 行


if not SKIP_VISION:
    vision_fixed = VisionEncoderFixed().eval()

    def pad_canvas(pv, pid):
        """把画布 pad 到固定 2520 patch（patch 补 0，position 补 -1）。"""
        P = pv.shape[1]
        assert P <= VISION_CANVAS
        pv = torch.nn.functional.pad(pv, (0, 0, 0, VISION_CANVAS - P))
        pid = torch.nn.functional.pad(pid, (0, 0, 0, VISION_CANVAS - P), value=-1)
        return pv, pid

    f0 = preproc_image(ASSETS / "red.png")
    pv0, pid0 = pad_canvas(f0["pixel_values"], f0["image_position_ids"])

    export(
        vision_fixed,
        (pv0, pid0),
        OUT_DIR / "vision_encoder_fixed.onnx",
        ["pixel_values", "image_position_ids"],
        ["image_embeds"],
        None,  # 完全静态
    )

    for name in ["red.png", "wide.png"]:
        f = preproc_image(ASSETS / name)
        n_valid = int((f["image_position_ids"] != -1).all(dim=-1).sum()) // 9
        with torch.no_grad():
            ref = vision_fixed(*pad_canvas(f["pixel_values"], f["image_position_ids"])).numpy()
        (out,) = ort_run(
            OUT_DIR / "vision_encoder_fixed.onnx",
            {
                "pixel_values": pad_canvas(f["pixel_values"], f["image_position_ids"])[0].numpy(),
                "image_position_ids": pad_canvas(f["pixel_values"], f["image_position_ids"])[1].numpy(),
            },
        )
        ok, msg = close(out[:n_valid], ref[:n_valid])
        # 再与动态版/裁剪版对照
        with torch.no_grad():
            ref_strip = vision(f["pixel_values"], f["image_position_ids"]).numpy()
        ok2, msg2 = close(ref[:n_valid], ref_strip)
        print(f"  vision_fixed[{name}]: onnx {msg} {'OK' if ok else 'FAIL'}; vs stripped {msg2} {'OK' if ok2 else 'FAIL'} (S={n_valid})")
        assert ok and ok2

# ============ 2) 音频编码器 ============
# 音频塔的 create_bidirectional_mask 内部用 vmap 构建掩码，两种导出器都处理不了。
# 导出期间 monkeypatch 成等价的广播实现（双向 + 左12/右0滑窗 + padding），
# 数值正确性由下面的 PyTorch 对照校验保证。
import transformers.models.gemma4.modeling_gemma4 as gemma4_modeling

_orig_create_bidirectional_mask = gemma4_modeling.create_bidirectional_mask


def _traceable_bidirectional_mask(config, inputs_embeds, attention_mask, **kwargs):
    B, T = inputs_embeds.shape[0], inputs_embeds.shape[1]
    device, dtype = inputs_embeds.device, inputs_embeds.dtype
    q = torch.arange(T, device=device).view(T, 1)
    kv = torch.arange(T, device=device).view(1, T)
    # sliding_window_mask_function((context_left-1=12, context_right=0))
    # 语义: dist = q-kv, 允许 0 <= dist < 12（含自身共 12 帧，不看未来）
    dist = q - kv
    allow = (dist >= 0) & (dist < 12)
    allow = allow.unsqueeze(0).expand(B, T, T)
    if attention_mask is not None:
        allow = allow & attention_mask[:, None, :].bool()
    # eager attention expects an additive float mask: 0 for valid, -inf for masked.
    min_dtype = torch.finfo(dtype).min
    return torch.where(
        allow,
        torch.tensor(0.0, dtype=dtype, device=device),
        torch.tensor(min_dtype, dtype=dtype, device=device),
    )[:, None]


class AudioEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_tower = backbone.audio_tower
        self.embed_audio = backbone.embed_audio

    def forward(self, input_features, input_features_mask):
        out = self.audio_tower(input_features, input_features_mask, return_dict=True)
        soft = self.embed_audio(out.last_hidden_state)  # [1, S, 512]
        return soft, out.attention_mask  # bool [1, S]


audio = AudioEncoder().eval()

# 先用原始掩码路径算参考值（此刻还没打补丁）
audio_refs = {}
for name in ["tone440.wav", "tone1000.wav"]:
    f = preproc_audio(ASSETS / name)
    with torch.no_grad():
        rs, rm = audio(f["input_features"], f["input_features_mask"])
    audio_refs[name] = (f, rs.numpy(), rm.numpy())

f_aud = audio_refs["tone440.wav"][0]
ref_soft, ref_mask = audio_refs["tone440.wav"][1], audio_refs["tone440.wav"][2]
n_placeholder = int((f_aud["input_ids"] == backbone.config.audio_token_id).sum())
print(f"audio soft tokens: {ref_soft.shape}, valid: {ref_mask.sum()}, placeholders: {n_placeholder}")
assert int(ref_mask.sum()) == n_placeholder

# 打补丁并验证：补丁版掩码的 PyTorch 输出必须与原始路径一致
# _extract_block_context 里的 F.pad+unfold 无法被旧版导出器处理，
# 用等价的 gather 实现替换（窗口起始位置都是静态计算的索引）。
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioAttention

_orig_extract_block_context = Gemma4AudioAttention._extract_block_context


def _traceable_extract_block_context(self, hidden_states):
    B, T, H, D = hidden_states.shape
    C, P, Fh, ctx = self.chunk_size, self.max_past_horizon, self.max_future_horizon, self.context_size
    nb = (T + C - 1) // C
    hp = torch.nn.functional.pad(hidden_states, (0, 0, 0, 0, P, Fh + C - 1))
    idx = torch.arange(ctx, device=hp.device).view(1, ctx) + torch.arange(nb, device=hp.device).view(nb, 1) * C
    return hp[:, idx.reshape(-1)].view(B, nb, ctx, H, D)


Gemma4AudioAttention._extract_block_context = _traceable_extract_block_context
gemma4_modeling.create_bidirectional_mask = _traceable_bidirectional_mask
with torch.no_grad():
    ps, pm = audio(f_aud["input_features"], f_aud["input_features_mask"])
ok, msg = close(ps.numpy(), ref_soft)
print(f"  patched mask vs original (pytorch): {msg} {'OK' if ok else 'FAIL'}")
assert ok and np.array_equal(pm.numpy(), ref_mask)

# 填充输入下也必须等价（之前只测未填充，漏掉了掩码 dtype 差异）
import torch.nn.functional as F

pad_n = 1000 - f_aud["input_features"].shape[1]
pf = F.pad(f_aud["input_features"], (0, 0, 0, pad_n))
pmask = F.pad(f_aud["input_features_mask"], (0, pad_n), value=False)
gemma4_modeling.create_bidirectional_mask = _orig_create_bidirectional_mask
with torch.no_grad():
    orig_pad = audio(pf, pmask)[0].numpy()
gemma4_modeling.create_bidirectional_mask = _traceable_bidirectional_mask
with torch.no_grad():
    patched_pad = audio(pf, pmask)[0].numpy()
ok, msg = close(patched_pad, orig_pad)
print(f"  patched vs original on PADDED input: {msg} {'OK' if ok else 'FAIL'}")
assert ok

# 音频按固定窗口导出：16kHz / hop160 下 1000 帧 ≈ 10 秒。
# 短音频 pad 到窗口（mask 置 0），长音频在 host 侧切成多个窗口后拼接软 token。
AUDIO_WINDOW_FRAMES = 1000


def pad_audio(feats: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    T = feats.shape[1]
    if T > AUDIO_WINDOW_FRAMES:
        raise ValueError(f"audio too long for one window: {T} > {AUDIO_WINDOW_FRAMES}")
    feats = torch.nn.functional.pad(feats, (0, 0, 0, AUDIO_WINDOW_FRAMES - T))
    mask = torch.nn.functional.pad(mask, (0, AUDIO_WINDOW_FRAMES - T), value=False)
    return feats, mask


padded_feats, padded_mask = pad_audio(f_aud["input_features"], f_aud["input_features_mask"])
padded_mask = padded_mask.float()  # ONNX 输入用 float32 掩码

export(
    audio,
    (padded_feats, padded_mask),
    OUT_DIR / "audio_encoder.onnx",
    ["input_features", "input_features_mask"],
    ["audio_embeds", "audio_valid_mask"],
    None,  # 静态 shape：音频按固定窗口导出，长音频在 host 侧分窗处理
    dynamo=False,  # 音频塔在 dynamo 导出器上会卡死，用旧版导出器
)

# 校验：两个音频 pad 到窗口长度后，有效 token 部分与 PyTorch 一致
for name in ["tone440.wav", "tone1000.wav"]:
    f, rs, rm = audio_refs[name]
    feats, mask = pad_audio(f["input_features"], f["input_features_mask"])
    with torch.no_grad():
        ref_soft, ref_mask = audio(feats, mask)
    out_soft, out_mask = ort_run(
        OUT_DIR / "audio_encoder.onnx",
        {"input_features": feats.numpy(), "input_features_mask": mask.numpy().astype(np.float32)},
    )
    n_valid = int(ref_mask.sum())
    ok1, msg = close(out_soft[:, :n_valid], ref_soft.numpy()[:, :n_valid])
    ok2 = np.array_equal(out_mask.astype(bool), ref_mask.numpy())
    print(f"  audio[{name}]: embeds(valid {n_valid}) {msg} {'OK' if ok1 else 'FAIL'}, mask {'OK' if ok2 else 'FAIL'}")
    assert ok1 and ok2

# 恢复原始掩码函数
gemma4_modeling.create_bidirectional_mask = _orig_create_bidirectional_mask

# ============ 3) 语言模型（inputs_embeds 版本）============
lm = backbone.language_model


class LMEmbeds(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lm = lm

    def forward(self, inputs_embeds, attention_mask):
        return self.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state


lm_wrap = LMEmbeds().eval()
dummy_embeds = torch.randn(1, 16, 512)
dummy_mask = torch.ones(1, 16, dtype=torch.int64)
with torch.no_grad():
    ref_lm = lm_wrap(dummy_embeds, dummy_mask).numpy()

export(
    lm_wrap,
    (dummy_embeds, dummy_mask),
    OUT_DIR / "lm_embeds.onnx",
    ["inputs_embeds", "attention_mask"],
    ["last_hidden_state"],
    {
        "inputs_embeds": {1: "sequence"},
        "attention_mask": {1: "sequence"},
    },
)

(out,) = ort_run(
    OUT_DIR / "lm_embeds.onnx",
    {"inputs_embeds": dummy_embeds.numpy(), "attention_mask": dummy_mask.numpy()},
)
ok, msg = close(out, ref_lm)  # 绝对误差
# LM 激活幅值可达 ~250（Gemma 特性），改用相对误差 + 逐 token 余弦相似度
rel = np.abs(out - ref_lm).max() / np.abs(ref_lm).max()
a = out.reshape(-1, 768)
b = ref_lm.reshape(-1, 768)
cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
ok = rel < 1e-3 and cos.min() > 0.9999
print(f"  lm_embeds: rel diff {rel:.2e}, min cos {cos.min():.6f} {'OK' if ok else 'FAIL'}")
assert ok

# ============ 4) 文本嵌入表 ============
table = backbone.language_model.embed_tokens.weight.detach().numpy().astype(bfloat16)
np.save(OUT_DIR / "token_embeddings.npy", table)
print("saved token_embeddings.npy:", table.shape, table.dtype)

# Restore monkeypatched attention helpers for any code that imports this tool.
gemma4_modeling.create_bidirectional_mask = _orig_create_bidirectional_mask
Gemma4AudioAttention._extract_block_context = _orig_extract_block_context

print("\nALL EXPORTS VERIFIED")
