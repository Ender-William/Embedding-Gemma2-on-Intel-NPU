"""Generate embedding_service/model/manifest.json with sizes and SHA256 hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = ROOT / "embedding_service" / "model"
MANIFEST = MODEL_ROOT / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hash", action="store_true", help="also compute SHA256")
    args = parser.parse_args()

    manifest = {"version": 1, "artifacts": {}}
    directories = ["onnx", "openvino"]
    quantized_root = MODEL_ROOT / "quantized"
    if quantized_root.exists():
        directories.extend(
            path.relative_to(MODEL_ROOT).as_posix()
            for path in sorted(quantized_root.iterdir())
            if path.is_dir() and (path / "model_info.json").exists()
        )
    for directory in directories:
        base = MODEL_ROOT / directory
        if not base.exists():
            continue
        files = {}
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            entry = {"size": path.stat().st_size}
            if args.hash:
                entry["sha256"] = sha256(path)
            files[path.relative_to(base).as_posix()] = entry
        manifest["artifacts"][directory] = files

    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {MANIFEST}")
    for name, files in manifest["artifacts"].items():
        total = sum(item["size"] for item in files.values()) / (1024**3)
        print(f"  {name}: {len(files)} files, {total:.2f} GiB")


if __name__ == "__main__":
    main()
