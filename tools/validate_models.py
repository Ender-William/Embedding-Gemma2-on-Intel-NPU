"""Validate local model packages against embedding_service/model/manifest.json."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = ROOT / "embedding_service" / "model"
MANIFEST = MODEL_ROOT / "manifest.json"

REQUIRED = {
    "onnx": {
        "embeddinggemma2_text.onnx",
        "model_info.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "config.json",
        "vision_encoder.onnx",
        "audio_encoder.onnx",
        "lm_embeds.onnx",
        "token_embeddings.npy",
    },
    "openvino": {
        "embeddinggemma2_text_openvino.xml",
        "model_info.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "config.json",
        "vision_encoder.xml",
        "audio_encoder.xml",
        "lm_embeds.xml",
        "token_embeddings.npy",
    },
}

QUANTIZED_REQUIRED = {
    "model_info.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "config.json",
    "vision_encoder.xml",
    "audio_encoder.xml",
    "lm_embeds.xml",
    "token_embeddings.npy",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hash", action="store_true", help="verify SHA256 entries")
    args = parser.parse_args()
    if not MANIFEST.exists():
        raise SystemExit(f"missing {MANIFEST}; run tools/build_manifest.py first")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    errors = []
    for name, files in manifest["artifacts"].items():
        dir_errors = []
        base = MODEL_ROOT / name
        required = (
            QUANTIZED_REQUIRED
            if name.startswith("quantized/")
            else REQUIRED.get(name, set())
        )
        missing_required = required - set(files)
        if missing_required:
            dir_errors.append(f"{name}: missing required entries {sorted(missing_required)}")
        for relative, entry in files.items():
            path = base / relative
            if not path.exists():
                dir_errors.append(f"missing file: {path}")
                continue
            if path.stat().st_size != entry["size"]:
                dir_errors.append(
                    f"size mismatch: {path} ({path.stat().st_size} != {entry['size']})"
                )
            if args.hash and "sha256" in entry and sha256(path) != entry["sha256"]:
                dir_errors.append(f"sha256 mismatch: {path}")
        errors.extend(dir_errors)
        print(f"{name}: {'OK' if not dir_errors else 'FAIL'}")

    if errors:
        raise SystemExit("\n".join(errors))
    print("all model packages valid")


if __name__ == "__main__":
    main()
