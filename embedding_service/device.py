"""Device capability and precision matrix."""

from __future__ import annotations

from .core import runtime_env  # noqa: F401 - register NVIDIA DLLs before ORT
from .types import DeviceUnavailableError, UnsupportedQuantizationError

DEVICES = ("cpu", "cuda", "intel_npu", "intel_gpu")
QUANTS = ("FP32", "FP16", "INT8", "INT8_W8", "INT8_FULL")


def normalize_device(device: str) -> str:
    value = device.strip().lower().replace("-", "_")
    aliases = {"npu": "intel_npu", "gpu": "intel_gpu", "intel_npu.0": "intel_npu"}
    value = aliases.get(value, value)
    if value not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {device!r}")
    return value


def normalize_quant(quant: str) -> str:
    value = quant.strip().upper().replace("-", "_")
    if value == "INT8":
        value = "INT8_W8"
    if value not in QUANTS:
        raise ValueError(f"quant must be one of {QUANTS}, got {quant!r}")
    return value


def resolve_backend(device: str, backend: str = "auto") -> str:
    value = backend.strip().lower()
    if value == "auto":
        return "onnxruntime" if device == "cuda" else "openvino"
    if value not in ("onnxruntime", "openvino"):
        raise ValueError("backend must be auto, onnxruntime, or openvino")
    if value == "onnxruntime" and device not in ("cpu", "cuda"):
        raise ValueError("onnxruntime supports device='cpu' or 'cuda'")
    if value == "openvino" and device == "cuda":
        raise ValueError("OpenVINO backend supports cpu/intel_npu/intel_gpu")
    return value


def validate_quantization(device: str, quant: str) -> str:
    device = normalize_device(device)
    quant = normalize_quant(quant)
    if device == "cuda" and quant.startswith("INT8"):
        raise UnsupportedQuantizationError(
            "CUDA INT8 is not supported by the current ONNX Runtime graph; "
            "use FP16/FP32 or OpenVINO intel devices"
        )
    return quant


def openvino_device_name(device: str) -> str:
    device = normalize_device(device)
    if device == "cpu":
        return "CPU"
    if device == "intel_npu":
        return "NPU"
    if device == "intel_gpu":
        return "GPU"
    raise ValueError("cuda is not an OpenVINO device")


def check_device_available(device: str) -> None:
    device = normalize_device(device)
    if device == "cpu":
        return
    if device == "cuda":
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise DeviceUnavailableError("onnxruntime-gpu is not installed") from exc
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" not in providers:
            raise DeviceUnavailableError("CUDAExecutionProvider is not available")
        return
    try:
        import openvino as ov
    except ImportError as exc:
        raise DeviceUnavailableError("openvino is not installed") from exc
    available = set(ov.Core().available_devices)
    wanted = "NPU" if device == "intel_npu" else "GPU"
    if not any(name == wanted or name.startswith(wanted + ".") for name in available):
        raise DeviceUnavailableError(f"{device} is not available; OpenVINO devices={sorted(available)}")
