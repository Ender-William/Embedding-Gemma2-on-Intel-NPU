"""Create model_info.json for complete model packages that exist under model/."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from embedding_service.manifest import build_file_manifest, write_json_atomic
from embedding_service.types import ModelArtifact

MODEL_ROOT = ROOT / "embedding_service" / "model"


def main() -> None:
    directories = [MODEL_ROOT / "onnx", MODEL_ROOT / "openvino"]
    quantized_root = MODEL_ROOT / "quantized"
    if quantized_root.exists():
        directories.extend(
            path for path in sorted(quantized_root.iterdir()) if path.is_dir()
        )

    registered = 0
    for directory in directories:
        if not directory.exists():
            continue
        onnx_text = sorted(
            path for path in directory.glob("*.onnx") if "text" in path.stem
        )
        openvino_text = sorted(
            path for path in directory.glob("*.xml") if "text" in path.stem
        )
        if onnx_text:
            backend = "onnxruntime"
            output_format = "onnx"
            text_path = onnx_text[0]
            quant = "FP32"
            device = "cuda"
            extension = ".onnx"
        elif openvino_text:
            backend = "openvino"
            output_format = "openvino_ir"
            text_path = openvino_text[0]
            quant = "INT8_W8" if "w8" in text_path.stem else "FP16"
            device = "intel_npu"
            extension = ".xml"
        else:
            print(f"skip {directory} (no text model)")
            continue

        components = {"text": str(text_path)}
        multi_modal = {}
        for key, filename in (
            ("vision", f"vision_encoder{extension}"),
            ("audio", f"audio_encoder{extension}"),
            ("lm", f"lm_embeds{extension}"),
            ("token_embeddings", "token_embeddings.npy"),
        ):
            path = directory / filename
            if path.exists():
                multi_modal[key] = str(path)
                if key != "token_embeddings":
                    components[key] = str(path)

        artifact = ModelArtifact(
            model_path=directory,
            source_model_path=MODEL_ROOT / "google--embeddinggemma-2",
            components=components,
            multi_modal=multi_modal,
            precision_policy="safe",
            manifest=directory / "model_info.json",
            validated=True,
            backend=backend,
            format=output_format,
            quant=quant,
            device=device,
        )
        write_json_atomic(
            directory / "model_info.json",
            {**artifact.to_dict(), "files": build_file_manifest(directory)},
        )
        print(f"registered {directory}")
        registered += 1
    if not registered:
        raise SystemExit(f"no complete model packages found under {MODEL_ROOT}")


if __name__ == "__main__":
    main()
