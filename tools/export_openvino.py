import shutil
from pathlib import Path

import openvino as ov

ROOT = Path(__file__).resolve().parent.parent
ONNX_PATH = ROOT / "embedding_service" / "model" / "onnx" / "embeddinggemma2_text.onnx"
ONNX_DIR = ROOT / "embedding_service" / "model" / "onnx"
OUT_DIR = ROOT / "embedding_service" / "model" / "openvino"
IR_PATH = OUT_DIR / "embeddinggemma2_text_openvino.xml"  # .bin 权重会自动放在同目录
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

# 1) ONNX -> OpenVINO IR（保留动态 shape，FP16 权重，体积减半）
#    IR 是设备无关的通用格式，分发给客户的就是这个文件
OUT_DIR.mkdir(parents=True, exist_ok=True)
model = core.read_model(str(ONNX_PATH))
ov.save_model(model, str(IR_PATH), compress_to_fp16=True)
print("saved IR:", IR_PATH)
for name in TOKENIZER_FILES:
    shutil.copy2(
        ROOT / "embedding_service" / "model" / "google--embeddinggemma-2" / name,
        OUT_DIR / name,
    )
print("tokenizer files copied to", OUT_DIR)

# 1.5) 多模态组件：固定形状的视觉编码器、音频编码器，以及 inputs_embeds 版语言模型
MM_FILES = {
    "vision_encoder_fixed.onnx": "vision_encoder.xml",  # [1,2520,768] -> [280,512]
    "audio_encoder.onnx": "audio_encoder.xml",          # [1,1000,128] -> [1,250,512]+mask
    "lm_embeds.onnx": "lm_embeds.xml",                  # 动态 [1,L,512] -> [1,L,768]
}
for onnx_name, ir_name in MM_FILES.items():
    src = ONNX_DIR / onnx_name
    if not src.exists():
        print(f"skip {onnx_name} (not found, run tools/export_multimodal_onnx.py first)")
        continue
    m = core.read_model(str(src))
    ov.save_model(m, str(OUT_DIR / ir_name), compress_to_fp16=True)
    print("saved IR:", ir_name)
shutil.copy2(ONNX_DIR / "token_embeddings.npy", OUT_DIR / "token_embeddings.npy")
print("multimodal artifacts copied")

