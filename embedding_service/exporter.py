"""ONNX/OpenVINO export and quantization helpers."""

from __future__ import annotations

import os
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

from .device import normalize_device, normalize_quant, openvino_device_name, resolve_backend
from .manifest import build_file_manifest, write_json_atomic
from .types import ModelArtifact, ModelArtifactError, UnsupportedQuantizationError

COMMON_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "config.json",
)

COMPONENT_NAMES = {
    "vision": "vision_encoder",
    "audio": "audio_encoder",
    "lm": "lm_embeds",
    "token_embeddings": "token_embeddings",
}

MULTIMODAL_ONNX_FILES = {
    "vision": "vision_encoder.onnx",
    "audio": "audio_encoder.onnx",
    "lm": "lm_embeds.onnx",
    "token_embeddings": "token_embeddings.npy",
}


def _is_hf_checkpoint(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").exists() and (
        (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()
    )


@contextmanager
def _utf8_console():
    """Keep Rich-based NNCF progress output usable on legacy GBK consoles."""
    states = []
    for stream in (sys.stdout, sys.stderr):
        if not hasattr(stream, "reconfigure"):
            continue
        try:
            states.append((stream, stream.encoding, stream.errors))
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            continue
    try:
        yield
    finally:
        for stream, encoding, errors in reversed(states):
            try:
                stream.reconfigure(encoding=encoding, errors=errors)
            except (AttributeError, OSError, ValueError):
                pass


def validate_multimodal_inputs(multi_modal: dict[str, str]) -> None:
    """Reject incomplete component sets before any expensive quantization starts."""
    if not multi_modal:
        return
    missing_keys = sorted(set(MULTIMODAL_ONNX_FILES) - set(multi_modal))
    if missing_keys:
        raise ModelArtifactError(
            "multimodal quantization requires all components; "
            f"missing: {missing_keys}"
        )
    unknown_keys = sorted(set(multi_modal) - set(MULTIMODAL_ONNX_FILES))
    if unknown_keys:
        raise ModelArtifactError(f"unknown multimodal components: {unknown_keys}")
    missing_paths = [
        f"{key}: {value}"
        for key, value in multi_modal.items()
        if not Path(value).expanduser().is_file()
    ]
    if missing_paths:
        raise ModelArtifactError(
            "multimodal component files do not exist before quantization:\n  "
            + "\n  ".join(missing_paths)
        )


def _copy_common(source: Path, output: Path) -> None:
    source_dir = source if source.is_dir() else source.parent
    for name in COMMON_FILES:
        candidate = source_dir / name
        if candidate.exists():
            shutil.copy2(candidate, output / name)


def _find_text_onnx(source_dir: Path) -> Path:
    candidates = sorted(source_dir.glob("*.onnx"))
    for path in candidates:
        if "text" in path.stem:
            return path
    if len(candidates) == 1:
        return candidates[0]
    raise ModelArtifactError(f"cannot identify text ONNX model in {source_dir}")


def _find_text_ir(source_dir: Path) -> Path:
    candidates = sorted(source_dir.glob("*.xml"))
    for path in candidates:
        if "text" in path.stem:
            return path
    if len(candidates) == 1:
        return candidates[0]
    raise ModelArtifactError(f"cannot identify text OpenVINO model in {source_dir}")


def export_text_onnx(model_path: Path, output: Path) -> Path:
    """Export the text backbone to ONNX using the already validated approach."""
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    from .core import runtime_env  # noqa: F401

    output.mkdir(parents=True, exist_ok=True)
    st = SentenceTransformer(
        str(model_path),
        device="cpu",
        model_kwargs={"dtype": "float32", "attn_implementation": "eager"},
    )
    lm = st[0].auto_model.language_model

    class TextEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm = lm

        def forward(self, input_ids, attention_mask):
            hidden = self.lm(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    wrapper = TextEncoder().eval()
    dummy_ids = torch.randint(3, 1000, (2, 16))
    dummy_mask = torch.ones_like(dummy_ids)
    onnx_path = output / "embeddinggemma2_text.onnx"
    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["sentence_embedding"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "sentence_embedding": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    _copy_common(model_path, output)
    return onnx_path


def _copy_onnx(source: Path, output: Path, target_name: str) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".onnx":
        destination = output / target_name
        shutil.copy2(source, destination)
        data = source.with_suffix(source.suffix + ".data")
        if data.exists():
            shutil.copy2(data, destination.with_suffix(destination.suffix + ".data"))
        return destination
    raise ModelArtifactError(f"expected an ONNX file, got {source}")


def _convert_onnx_fp16(source: Path, destination: Path) -> Path:
    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError as exc:
        raise UnsupportedQuantizationError(
            "FP16 ONNX conversion requires onnxconverter-common"
        ) from exc
    model = onnx.load(str(source))
    converted = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save_model(
        converted,
        str(destination),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=destination.name + ".data",
        size_threshold=1024,
    )
    return destination


def _prepare_onnx(
    source: Path,
    staging: Path,
    multi_modal: dict[str, str],
    quant: str,
) -> Path:
    if _is_hf_checkpoint(source):
        text_onnx = export_text_onnx(source, staging)
    else:
        source_dir = source if source.is_dir() else source.parent
        text_onnx = _find_text_onnx(source_dir)
        shutil.copy2(text_onnx, staging / "embeddinggemma2_text.onnx")
        external = text_onnx.with_suffix(text_onnx.suffix + ".data")
        if external.exists():
            shutil.copy2(
                external,
                staging / "embeddinggemma2_text.onnx.data",
            )
        _copy_common(source, staging)

    if quant == "FP16":
        converted = staging / "embeddinggemma2_text_fp16.onnx"
        _convert_onnx_fp16(staging / "embeddinggemma2_text.onnx", converted)
        converted.replace(staging / "embeddinggemma2_text.onnx")

    for key, source_value in multi_modal.items():
        if key not in COMPONENT_NAMES:
            raise ValueError(f"unknown multimodal component {key!r}")
        src = Path(source_value)
        if key == "token_embeddings":
            shutil.copy2(src, staging / "token_embeddings.npy")
            continue
        if src.suffix != ".onnx":
            raise ModelArtifactError(f"{key} must point to an ONNX file")
        target = staging / f"{COMPONENT_NAMES[key]}.onnx"
        shutil.copy2(src, target)
        data = src.with_suffix(src.suffix + ".data")
        if data.exists():
            shutil.copy2(data, target.with_suffix(target.suffix + ".data"))
        if quant == "FP16":
            converted = target.with_suffix(".fp16.onnx")
            _convert_onnx_fp16(target, converted)
            converted.replace(target)

    if multi_modal and not (staging / "token_embeddings.npy").exists():
        raise ModelArtifactError("multimodal ONNX package requires token_embeddings")
    return _find_text_onnx(staging)


def _save_ir(model, output: Path, quant: str) -> None:
    import openvino as ov

    if quant.startswith("INT8"):
        import nncf

        with _utf8_console():
            model = nncf.compress_weights(
                model,
                mode=nncf.CompressWeightsMode.INT8_ASYM,
            )
    ov.save_model(model, str(output), compress_to_fp16=(quant == "FP16"))


def _prepare_openvino(
    source: Path,
    staging: Path,
    multi_modal: dict[str, str],
    quant: str,
    cache_root: Path | None,
) -> Path:
    import openvino as ov

    from .core.cache import temporary_compile_cache

    core = ov.Core()
    text_output = staging / (
        "embeddinggemma2_text_w8.xml"
        if quant.startswith("INT8")
        else "embeddinggemma2_text_openvino.xml"
    )

    has_onnx = source.is_dir() and any(source.glob("*.onnx"))
    has_xml = source.is_dir() and any(source.glob("*.xml"))
    if source.suffix == ".onnx" or _is_hf_checkpoint(source) or has_onnx:
        if _is_hf_checkpoint(source):
            onnx_dir = staging / "_onnx"
            text_onnx = export_text_onnx(source, onnx_dir)
        else:
            text_onnx = _find_text_onnx(source if source.is_dir() else source.parent)
        model = core.read_model(str(text_onnx))
        _save_ir(model, text_output, quant)
    elif source.suffix == ".xml" or has_xml:
        text_ir = _find_text_ir(source if source.is_dir() else source.parent)
        model = core.read_model(str(text_ir))
        _save_ir(model, text_output, quant)
    else:
        raise ModelArtifactError(f"unsupported source model: {source}")

    _copy_common(source, staging)
    for key, source_value in multi_modal.items():
        if key not in COMPONENT_NAMES:
            raise ValueError(f"unknown multimodal component {key!r}")
        src = Path(source_value)
        if key == "token_embeddings":
            shutil.copy2(src, staging / "token_embeddings.npy")
            continue
        output_name = f"{COMPONENT_NAMES[key]}.xml"
        if src.suffix == ".onnx":
            model = core.read_model(str(src))
        elif src.suffix == ".xml":
            model = core.read_model(str(src))
        else:
            raise ModelArtifactError(f"{key} must point to ONNX or OpenVINO IR")
        _save_ir(model, staging / output_name, quant)
    if multi_modal and not (staging / "token_embeddings.npy").exists():
        raise ModelArtifactError("multimodal OpenVINO package requires token_embeddings")
    shutil.rmtree(staging / "_onnx", ignore_errors=True)
    return text_output


def quantize_model(
    model_path: str | Path,
    multi_modal: dict[str, str] | None,
    quant: str,
    device: str,
    *,
    backend: str = "auto",
    output_dir: str | Path | None = None,
    calibration_data: str | Path | None = None,
    overwrite: bool = False,
    cache_root: str | Path | None = None,
) -> ModelArtifact:
    source = Path(model_path).expanduser().resolve()
    if not source.exists():
        raise ModelArtifactError(f"model path does not exist: {source}")
    device = normalize_device(device)
    quant = normalize_quant(quant)
    backend_name = resolve_backend(device, backend)
    if quant == "INT8_FULL" and calibration_data is None:
        raise UnsupportedQuantizationError(
            "INT8_FULL requires calibration_data; use INT8_W8 for weight-only quantization"
        )
    if quant == "INT8_FULL":
        raise UnsupportedQuantizationError(
            "INT8_FULL is not implemented yet; use INT8_W8"
        )
    if device == "cuda" and quant.startswith("INT8"):
        raise UnsupportedQuantizationError("CUDA INT8 is not supported")

    multi_modal = {
        k: str(Path(v).expanduser().resolve())
        for k, v in (multi_modal or {}).items()
    }
    validate_multimodal_inputs(multi_modal)
    stem = source.stem if source.is_file() else source.name
    if output_dir is None:
        output_dir = source.parent / "quantized" / f"{stem}__{quant.lower()}__{device}"
    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"quantized output already exists: {target}")
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    staging = target.parent / f".tmp-{target.name}-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        if backend_name == "onnxruntime":
            text_path = _prepare_onnx(source, staging, multi_modal, quant)
            output_format = "onnx"
            precision_policy = "fp16_io_guarded" if quant == "FP16" else "fp32"
        else:
            text_path = _prepare_openvino(
                source, staging, multi_modal, quant, Path(cache_root) if cache_root else None
            )
            output_format = "openvino_ir"
            precision_policy = (
                "npu_fp16_with_precision_guard"
                if device == "intel_npu" and quant == "FP16"
                else "fp32_activations"
                if quant.startswith("INT8")
                else "fp16_weights_fp32_activations"
            )

        artifact = ModelArtifact(
            model_path=target,
            source_model_path=source,
            backend=backend_name,
            format=output_format,
            quant=quant,
            device=device,
            components={"text": str(text_path)},
            multi_modal=multi_modal,
            precision_policy=precision_policy,
            validated=True,
        )
        manifest_path = staging / "model_info.json"
        artifact.manifest = target / "model_info.json"
        artifact.components = {
            key: str(target / Path(value).name)
            for key, value in artifact.components.items()
        }
        if multi_modal:
            suffix = ".onnx" if output_format == "onnx" else ".xml"
            artifact.multi_modal = {
                key: str(
                    target
                    / (
                        "token_embeddings.npy"
                        if key == "token_embeddings"
                        else f"{COMPONENT_NAMES[key]}{suffix}"
                    )
                )
                for key in multi_modal
            }
            for key, value in artifact.multi_modal.items():
                if key != "token_embeddings":
                    artifact.components[key] = value
        write_json_atomic(
            manifest_path,
            {
                **artifact.to_dict(),
                "model_path": str(target),
                "manifest": str(target / "model_info.json"),
                "files": build_file_manifest(staging),
            },
        )
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    artifact.model_path = target
    artifact.manifest = target / "model_info.json"
    return artifact
