import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from embedding_service.core import runtime_env  # noqa: E402,F401
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_DIR = ROOT / "embedding_service" / "model" / "google--embeddinggemma-2"
OUT_DIR = ROOT / "embedding_service" / "model" / "onnx"
ONNX_PATH = OUT_DIR / "embeddinggemma2_text.onnx"
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "config.json",
]

st = SentenceTransformer(
    str(MODEL_DIR),
    device="cpu",
    model_kwargs={"dtype": "float32", "attn_implementation": "eager"},
)
lm = st[0].auto_model.language_model


class TextEncoder(torch.nn.Module):
    """input_ids + attention_mask -> mean pooling -> 768d embedding（归一化在 numpy 侧做，NPU 上更稳）"""

    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def forward(self, input_ids, attention_mask):
        hidden = self.lm(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


wrapper = TextEncoder(lm).eval()

dummy_ids = torch.randint(3, 1000, (2, 16))
dummy_mask = torch.ones_like(dummy_ids)

with torch.no_grad():
    ref = wrapper(dummy_ids, dummy_mask)

OUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask),
        str(ONNX_PATH),
        input_names=["input_ids", "attention_mask"],
        output_names=["sentence_embedding"],
        dynamic_shapes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
        },
        opset_version=17,
        dynamo=True,
    )
except Exception as e:
    print("dynamo export failed, falling back to legacy exporter:", e)
    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask),
        str(ONNX_PATH),
        input_names=["input_ids", "attention_mask"],
        output_names=["sentence_embedding"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "embedding": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
print("exported:", ONNX_PATH)

# 分词器一并放入模型目录，使其成为可分发的完整单元
for name in TOKENIZER_FILES:
    shutil.copy2(MODEL_DIR / name, OUT_DIR / name)
print("tokenizer files copied to", OUT_DIR)

# 用 onnxruntime 校验导出结果与 PyTorch 一致
import onnxruntime as ort

sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
out = sess.run(
    [sess.get_outputs()[0].name],
    {"input_ids": dummy_ids.numpy(), "attention_mask": dummy_mask.numpy()},
)[0]
ref_n = ref.numpy() / np.linalg.norm(ref.numpy(), axis=1, keepdims=True)
out_n = out / np.linalg.norm(out, axis=1, keepdims=True)
diff = np.abs(out_n - ref_n).max()
print("max abs diff vs pytorch (normalized):", diff)
assert diff < 1e-3, "export verification failed"
print("OK")
