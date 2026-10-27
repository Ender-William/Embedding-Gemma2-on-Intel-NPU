"""EmbeddingService end-to-end demo.

Examples:
    uv run python examples/demo.py list
    uv run python examples/demo.py demo --mode all
    uv run python examples/demo.py demo --device intel_npu --mode all
    uv run python examples/demo.py download --repo google/embeddinggemma-2 --source hf-mirror
    uv run python examples/demo.py quantize --model-path ... --quant INT8 --device intel_npu
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
import wave
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

import av
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from embedding_service import (  # noqa: E402
    DEFAULT_MODEL_ROOT,
    EmbeddingService,
    ModelArtifact,
)

ASSETS = ROOT / "examples" / "assets"


def ensure_assets() -> dict[str, Path]:
    ASSETS.mkdir(parents=True, exist_ok=True)
    colors = {
        "red": (220, 30, 30),
        "blue": (30, 30, 220),
        "green": (30, 200, 70),
    }
    for name, color in colors.items():
        path = ASSETS / f"{name}.png"
        if not path.exists():
            Image.new("RGB", (640, 640), color).save(path)
    wide = ASSETS / "wide.png"
    if not wide.exists():
        Image.new("RGB", (1280, 480), (210, 160, 30)).save(wide)

    for name, frequency in (("tone440.wav", 440.0), ("tone1000.wav", 1000.0)):
        path = ASSETS / name
        if path.exists():
            continue
        rate = 16000
        samples = [
            int(12000 * math.sin(2 * math.pi * frequency * t / rate))
            for t in range(rate * 2)
        ]
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(struct.pack(f"<{len(samples)}h", *samples))

    for name, transitions in (
        ("red_blue_3s.mp4", ((220, 30, 30), (30, 30, 220))),
        ("green_blue_3s.mp4", ((30, 200, 70), (30, 30, 220))),
    ):
        path = ASSETS / name
        if not path.exists():
            _make_video(path, transitions)

    return {
        "red": ASSETS / "red.png",
        "blue": ASSETS / "blue.png",
        "green": ASSETS / "green.png",
        "wide": ASSETS / "wide.png",
        "tone440": ASSETS / "tone440.wav",
        "tone1000": ASSETS / "tone1000.wav",
        "video_red_blue": ASSETS / "red_blue_3s.mp4",
        "video_green_blue": ASSETS / "green_blue_3s.mp4",
    }


def _make_video(path: Path, transitions) -> None:
    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=5)
    stream.width = 320
    stream.height = 240
    stream.pix_fmt = "yuv420p"
    frames = []
    for color in transitions:
        frames.extend([color] * 7)
    for color in frames:
        array = np.empty((240, 320, 3), dtype=np.uint8)
        array[:] = color
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def default_device() -> str:
    try:
        import onnxruntime as ort

        if "CUDAExecutionProvider" in ort.get_available_providers():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def create_service(args) -> EmbeddingService:
    return EmbeddingService(
        models_root=args.models_root,
        cache_root=args.cache_root,
        default_device=args.device or default_device(),
        precision_policy="safe",
        verbose=not args.quiet,
    )


def add_service_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "cache")
    parser.add_argument("--device", choices=["cpu", "cuda", "intel_npu", "intel_gpu"])
    parser.add_argument("--quiet", action="store_true")


def select_artifact(
    service: EmbeddingService,
    selector: str | None,
    device: str,
    backend: str | None,
    quant: str | None,
) -> ModelArtifact:
    models = service.list_models()
    if not models:
        raise SystemExit(
            f"no registered models under {service.models_root}; "
            "run `python tools/register_existing_models.py`"
        )
    if selector:
        matches = [
            item
            for item in models
            if selector in (item.model_path.name, str(item.model_path))
        ]
        if not matches:
            raise SystemExit(f"no model matches {selector!r}")
        return matches[0]

    preferred_backend = backend or ("onnxruntime" if device == "cuda" else "openvino")
    candidates = [
        item for item in models if item.backend == preferred_backend
    ] or models
    if quant:
        normalized = quant.upper()
        candidates = [
            item for item in candidates if item.quant == normalized
        ] or candidates
    if device == "intel_npu":
        candidates.sort(key=lambda item: item.quant != "INT8_W8")
    return candidates[0]


def ranking(
    service: EmbeddingService,
    query: str,
    documents: list,
    labels: list[str],
    prompt: str | None = None,
) -> None:
    query_emb = service.run([query], prompt="query").embeddings
    doc_emb = service.run(documents, prompt=prompt).embeddings
    scores = service.similarity(doc_emb, query_emb)[:, 0]
    for index in scores.argsort()[::-1]:
        print(f"  {scores[index]:.4f}  {labels[index]}")


def print_result(label: str, result) -> None:
    print(
        f"{label}: shape={result.embeddings.shape}, "
        f"backend={result.backend}, device={result.device}, "
        f"quant={result.quant}, latency={result.elapsed_ms:.1f} ms"
    )


def run_demo(args) -> int:
    assets = ensure_assets()
    service = create_service(args)
    device = args.device or default_device()
    artifact = select_artifact(service, args.model, device, args.backend, args.quant)
    print(f"selected model: {artifact.model_path}")
    print(f"target device: {device}")
    try:
        service.load(artifact, device=device, warmup=True)
        print(f"service status: {json.dumps(service.status(), ensure_ascii=False)}")
        modes = ["text", "image", "audio", "video", "mixed"] if args.mode == "all" else [args.mode]

        if "text" in modes:
            print("\n[text]")
            documents = [
                "木星是太阳系中体积最大的行星。",
                "今天晚饭吃什么好呢？",
                "Python 是一种流行的编程语言。",
            ]
            result = service.run(documents, prompt="document")
            print_result("text embeddings", result)
            ranking(
                service,
                "太阳系中最大的行星是哪一颗？",
                documents,
                [item[:30] for item in documents],
                prompt="document",
            )

        available = set(artifact.multi_modal)
        if "image" in modes and {"vision", "lm", "token_embeddings"} <= available:
            print("\n[image]")
            image_docs = [
                {"text": "<|image|>", "image": [str(assets["red"])]},
                {"text": "<|image|>", "image": [str(assets["blue"])]},
                {"text": "<|image|>", "image": [str(assets["green"])]},
            ]
            ranking(
                service,
                "一张红色的方块图片",
                image_docs,
                ["red", "blue", "green"],
            )

        if "audio" in modes and {"audio", "lm", "token_embeddings"} <= available:
            print("\n[audio]")
            audio_docs = [
                {"text": "<|audio|>", "audio": [str(assets["tone440"])]},
                {"text": "<|audio|>", "audio": [str(assets["tone1000"])]},
            ]
            ranking(service, "一段蜂鸣音", audio_docs, ["440 Hz", "1000 Hz"])

        if "video" in modes and {"vision", "lm", "token_embeddings"} <= available:
            print("\n[video]")
            video_docs = [
                {"text": "<|video|>", "video": str(assets["video_red_blue"])},
                {"text": "<|video|>", "video": str(assets["video_green_blue"])},
            ]
            ranking(
                service,
                "视频从红色变化到蓝色",
                video_docs,
                ["red -> blue", "green -> blue"],
            )

        if "mixed" in modes and {"vision", "lm", "token_embeddings"} <= available:
            print("\n[mixed]")
            mixed_docs = [
                {
                    "text": "这是一张红色图片 <|image|>",
                    "image": [str(assets["red"])],
                },
                {
                    "text": "这是一张绿色图片 <|image|>",
                    "image": [str(assets["green"])],
                },
            ]
            ranking(service, "红色图片", mixed_docs, ["red + text", "green + text"])

        print(f"\nfinal status: {json.dumps(service.status(), ensure_ascii=False)}")
        return 0
    finally:
        service.unload()


def run_download(args) -> int:
    service = create_service(args)
    result = service._download_model(
        repo_name=args.repo,
        source=args.source,
        revision=args.revision,
        force=args.force,
        max_workers=args.max_workers,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


def run_quantize(args) -> int:
    service = create_service(args)
    explicit = {
        "vision": args.vision,
        "audio": args.audio,
        "lm": args.lm,
        "token_embeddings": args.token_embeddings,
    }
    if args.text_only:
        if any(explicit.values()):
            raise SystemExit(
                "--text-only cannot be combined with multimodal component paths"
            )
        multi_modal = None
    else:
        default_dir = args.models_root / "onnx"
        multi_modal = {
            "vision": str(args.vision or default_dir / "vision_encoder.onnx"),
            "audio": str(args.audio or default_dir / "audio_encoder.onnx"),
            "lm": str(args.lm or default_dir / "lm_embeds.onnx"),
            "token_embeddings": str(
                args.token_embeddings or default_dir / "token_embeddings.npy"
            ),
        }
    artifact = service._quantization(
        model_path=args.model_path,
        multi_modal=multi_modal,
        quant=args.quant,
        device=args.device or "intel_npu",
        backend=args.backend,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2))
    return 0


def run_list(args) -> int:
    service = create_service(args)
    models = service.list_models()
    if not models:
        print("No registered models.")
        return 1
    for artifact in models:
        print(
            f"{artifact.model_path.relative_to(service.models_root)}\n"
            f"  backend={artifact.backend} device={artifact.device} "
            f"quant={artifact.quant} multimodal={sorted(artifact.multi_modal)}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EmbeddingService demo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list registered models")
    add_service_arguments(list_parser)
    list_parser.set_defaults(func=run_list)

    demo_parser = subparsers.add_parser("demo", help="run multimodal retrieval demo")
    add_service_arguments(demo_parser)
    demo_parser.add_argument("--model", help="model directory name or full path")
    demo_parser.add_argument("--backend", choices=["onnxruntime", "openvino"])
    demo_parser.add_argument("--quant", choices=["FP32", "FP16", "INT8", "INT8_W8"])
    demo_parser.add_argument(
        "--mode",
        choices=["all", "text", "image", "audio", "video", "mixed"],
        default="all",
    )
    demo_parser.set_defaults(func=run_demo)

    download_parser = subparsers.add_parser("download", help="download a HuggingFace model")
    add_service_arguments(download_parser)
    download_parser.add_argument("--repo", required=True)
    download_parser.add_argument(
        "--source", choices=["huggingface", "hf-mirror"], default="huggingface"
    )
    download_parser.add_argument("--revision")
    download_parser.add_argument("--force", action="store_true")
    download_parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="parallel file downloads; keep at 1 for hf-mirror",
    )
    download_parser.set_defaults(func=run_download)

    quantize_parser = subparsers.add_parser("quantize", help="quantize a model")
    add_service_arguments(quantize_parser)
    quantize_parser.add_argument("--model-path", type=Path, required=True)
    quantize_parser.add_argument("--quant", choices=["FP32", "FP16", "INT8", "INT8_W8"], default="INT8")
    quantize_parser.add_argument("--backend", choices=["auto", "onnxruntime", "openvino"], default="auto")
    quantize_parser.add_argument("--output-dir", type=Path)
    quantize_parser.add_argument("--overwrite", action="store_true")
    quantize_parser.add_argument("--vision", type=Path)
    quantize_parser.add_argument("--audio", type=Path)
    quantize_parser.add_argument("--lm", type=Path)
    quantize_parser.add_argument("--token-embeddings", type=Path)
    quantize_parser.add_argument(
        "--text-only",
        action="store_true",
        help="quantize only the text model instead of the full multimodal package",
    )
    quantize_parser.set_defaults(func=run_quantize)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
