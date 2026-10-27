"""用 NNCF 对文本编码器做 INT8 权重量化（W8），产出通用 OpenVINO IR。

权重量化不需要校准数据：权重压成 INT8，激活保持 FP16/FP32，
体积减半，精度损失通常可以忽略。
"""

import nncf
import openvino as ov
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ONNX_PATH = ROOT / "embedding_service" / "model" / "onnx" / "embeddinggemma2_text.onnx"
OUT_DIR = (
    ROOT
    / "embedding_service"
    / "model"
    / "quantized"
    / "embeddinggemma-2__int8__intel_npu"
)
W8_IR_PATH = OUT_DIR / "embeddinggemma2_text_w8.xml"
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "config.json",
]

core = ov.Core()
model = core.read_model(str(ONNX_PATH))

compressed = nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT8_ASYM)

OUT_DIR.mkdir(parents=True, exist_ok=True)
ov.save_model(compressed, str(W8_IR_PATH))
print("saved W8 IR:", W8_IR_PATH)

# 多模态组件也做 W8（若存在），并复制配套文件
MM_FILES = {
    "vision_encoder.xml": "vision_encoder_fixed.onnx",
    "audio_encoder.xml": "audio_encoder.onnx",
    "lm_embeds.xml": "lm_embeds.onnx",
}
FP16_DIR = ROOT / "embedding_service" / "model" / "openvino"
for ir_name, onnx_name in MM_FILES.items():
    src = ONNX_PATH.parent / onnx_name
    if not src.exists():
        continue
    m = core.read_model(str(src))
    ov.save_model(
        nncf.compress_weights(m, mode=nncf.CompressWeightsMode.INT8_ASYM),
        str(OUT_DIR / ir_name),
    )
    print("saved W8 IR:", ir_name)

if (ONNX_PATH.parent / "token_embeddings.npy").exists():
    shutil.copy2(ONNX_PATH.parent / "token_embeddings.npy", OUT_DIR / "token_embeddings.npy")

for name in TOKENIZER_FILES:
    shutil.copy2(
        ROOT / "embedding_service" / "model" / "google--embeddinggemma-2" / name,
        OUT_DIR / name,
    )
print("tokenizer files copied to", OUT_DIR)
