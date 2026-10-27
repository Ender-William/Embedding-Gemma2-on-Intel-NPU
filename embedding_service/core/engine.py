r"""State-machine Embedding engine for ONNX Runtime and OpenVINO.

    INIT -> PROBING -> BACKEND_SELECTED -> LOADING -> READY
       \         \            \               \
        ----------->---------->------ FAILED <--

The LOADING state can return to BACKEND_SELECTED when a preferred backend
fails to initialize, allowing the next available backend to be tried.
"""

from __future__ import annotations

import os
from enum import Enum, auto
from pathlib import Path
from typing import Iterable

import numpy as np

from . import runtime_env  # noqa: F401 - register bundled NVIDIA DLLs before ORT
from .cache import temporary_compile_cache

MODEL_ROOT = Path(__file__).resolve().parents[1] / "model"
MODEL_MAX_SEQUENCE = 8192
DEFAULT_OPENVINO_SEQ_LEN = 512

PROMPTS = {
    "query": "task: search result | query: ",
    "document": "title: none | text: ",
}

MM_ONNX = ("vision_encoder.onnx", "audio_encoder.onnx", "lm_embeds.onnx", "token_embeddings.npy")
MM_OV = ("vision_encoder.xml", "audio_encoder.xml", "lm_embeds.xml", "token_embeddings.npy")
QUANTIZED_DIR_NAME = "embeddinggemma-2__int8__intel_npu"


class EngineState(Enum):
    INIT = auto()
    PROBING = auto()
    BACKEND_SELECTED = auto()
    LOADING = auto()
    READY = auto()
    FAILED = auto()


_TRANSITIONS = {
    EngineState.INIT: {EngineState.PROBING, EngineState.FAILED},
    EngineState.PROBING: {EngineState.BACKEND_SELECTED, EngineState.FAILED},
    EngineState.BACKEND_SELECTED: {EngineState.LOADING, EngineState.FAILED},
    EngineState.LOADING: {EngineState.READY, EngineState.BACKEND_SELECTED, EngineState.FAILED},
    EngineState.READY: set(),
    EngineState.FAILED: set(),
}


def _artifact_status(model_dir: Path, backend: str) -> str:
    """Return 'full', 'none', or 'partial' for optional multimodal artifacts."""
    names = MM_ONNX if backend == "onnx" else MM_OV
    present = [name for name in names if (model_dir / name).exists()]
    if len(present) == len(names):
        return "full"
    if not present:
        return "none"
    missing = [name for name in names if name not in present]
    raise RuntimeError(f"partial multimodal package in {model_dir}; missing: {missing}")


def _normalize_embedding(embeds: np.ndarray, name: str) -> np.ndarray:
    embeds = np.asarray(embeds, dtype=np.float32)
    if not np.isfinite(embeds).all():
        raise RuntimeError(f"{name} produced non-finite embeddings")
    norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    if np.any(norms <= 1e-8):
        raise RuntimeError(f"{name} produced zero-length embeddings")
    return embeds / norms


def _with_prompt(texts: list[str], prompt: str | None) -> list[str]:
    if prompt is None:
        return texts
    if prompt not in PROMPTS:
        raise ValueError(f"unknown prompt {prompt!r}; expected {sorted(PROMPTS)}")
    return [PROMPTS[prompt] + text for text in texts]


class _OnnxBackend:
    name = "onnx"

    def __init__(
        self,
        model_dir: Path,
        max_text_length: int = MODEL_MAX_SEQUENCE,
        device: str | None = None,
        **_,
    ):
        self.model_dir = model_dir
        self.max_text_length = max_text_length
        self.device_preference = device
        self.mm = None
        self.status = "missing"

    def available(self) -> bool:
        if not (self.model_dir / "embeddinggemma2_text.onnx").exists():
            self.status = "missing text model"
            return False
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            self.status = "onnxruntime not installed"
            return False
        try:
            self.status = _artifact_status(self.model_dir, "onnx")
        except RuntimeError as exc:
            self.status = str(exc)
            return False
        return True

    def load(self):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        providers = (
            ["CPUExecutionProvider"]
            if self.device_preference == "cpu"
            else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(
            str(self.model_dir / "embeddinggemma2_text.onnx"),
            providers=providers,
        )
        self.out_name = self.session.get_outputs()[0].name
        self.device = (
            "cuda" if "CUDAExecutionProvider" in self.session.get_providers() else "cpu"
        )
        if self.status == "full":
            from .multimodal import OnnxMultimodalEncoder

            mm_providers = (
                ["CPUExecutionProvider"]
                if self.device_preference == "cpu"
                else ["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            self.mm = OnnxMultimodalEncoder(
                self.model_dir,
                providers=mm_providers,
            )

    def encode(self, inputs, prompt: str | None = None) -> np.ndarray:
        if any(isinstance(item, dict) for item in inputs):
            if self.mm is None:
                raise RuntimeError(f"multimodal artifacts are not complete in {self.model_dir}")
            return self.mm.encode(inputs, prompt=prompt)
        texts = _with_prompt(list(inputs), prompt)
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="np",
        )
        embs = self.session.run(
            [self.out_name],
            {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            },
        )[0]
        return _normalize_embedding(embs, "onnx")


class _OpenVinoBackend:
    name = "openvino"

    def __init__(
        self,
        model_dir: Path,
        seq_len: int = DEFAULT_OPENVINO_SEQ_LEN,
        max_text_length: int = MODEL_MAX_SEQUENCE,
        device: str | None = None,
        precision_check: bool = True,
        cache_root: str | Path | None = None,
        **_,
    ):
        self.model_dir = model_dir
        self.seq_len = seq_len
        self.max_text_length = max_text_length
        self.device_preference = device
        self.precision_check = precision_check
        self.cache_root = cache_root
        self.mm = None
        self.safe_session = None
        self.static_session = None
        self.precision_fallback = False
        self.fallback_reason = ""
        self.status = "missing"
        xmls = sorted(self.model_dir.glob("*.xml"))
        self.ir_path = next((p for p in xmls if "text" in p.name), xmls[0] if xmls else None)

    def available(self) -> bool:
        if self.ir_path is None:
            self.status = "missing text IR"
            return False
        try:
            import openvino  # noqa: F401
        except ImportError:
            self.status = "openvino not installed"
            return False
        try:
            self.status = _artifact_status(self.model_dir, "openvino")
        except RuntimeError as exc:
            self.status = str(exc)
            return False
        return True

    def _compile(self, device: str, static: bool):
        import openvino as ov

        model = self.core.read_model(str(self.ir_path))
        if static:
            model.reshape(
                {"input_ids": [1, self.seq_len], "attention_mask": [1, self.seq_len]}
            )
        precision = "f16" if device.upper() == "NPU" else "f32"
        config = {"INFERENCE_PRECISION_HINT": precision}
        with temporary_compile_cache(self.cache_root) as cache_dir:
            if cache_dir is not None:
                config["CACHE_DIR"] = cache_dir
            return self.core.compile_model(model, device, config)

    def load(self):
        import openvino as ov
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.core = ov.Core()
        devices = list(self.core.available_devices)
        requested = (self.device_preference or ("NPU" if "NPU" in devices else "CPU")).upper()
        if requested == "NPU" and "NPU" not in devices:
            requested = "CPU"
        self.device = requested.lower()
        self.safe_session = self._compile("CPU", static=False)
        if requested == "NPU":
            try:
                self.static_session = self._compile("NPU", static=True)
                if self.precision_check:
                    self._run_precision_check()
            except Exception as exc:
                self.static_session = None
                self.device = "cpu"
                self.fallback_reason = f"NPU text model unavailable: {exc}"
        else:
            self.static_session = None

        if self.status == "full":
            from .multimodal import OpenVinoMultimodalEncoder

            dev = "NPU" if self.device == "npu" else "CPU"
            try:
                self.mm = OpenVinoMultimodalEncoder(
                    self.model_dir,
                    device={"vision": "CPU", "audio": dev, "lm": dev},
                    seq_len=max(self.seq_len, 1024),
                    cache_root=self.cache_root,
                )
            except Exception as exc:
                self.fallback_reason = f"NPU multimodal unavailable: {exc}"
                self.mm = OpenVinoMultimodalEncoder(
                    self.model_dir,
                    device="CPU",
                    seq_len=max(self.seq_len, 1024),
                    cache_root=self.cache_root,
                )

    def _tokenize(self, texts: list[str], max_length: int, pad_to_max: bool):
        return self.tokenizer(
            texts,
            padding="max_length" if pad_to_max else True,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )

    def _run_session(
        self,
        session,
        texts: list[str],
        max_length: int | None = None,
        pad_to_max: bool = False,
    ) -> np.ndarray:
        enc = self._tokenize(texts, max_length or self.max_text_length, pad_to_max)
        embs = session(
            {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            }
        )
        return _normalize_embedding(
            np.asarray(embs[session.output(0)], dtype=np.float32), "openvino"
        )

    def _run_precision_check(self):
        samples = [
            "太阳系中最大的行星是哪一颗？",
            "a red square image",
            "如何排查网络连接故障？",
        ]
        npu = np.stack([
            self._run_session(self.static_session, [sample], self.seq_len, pad_to_max=True)[0]
            for sample in samples
        ])
        cpu = self._run_session(self.safe_session, samples)
        cos = (npu * cpu).sum(axis=1)
        if not np.isfinite(cos).all() or float(cos.min()) < 0.995:
            self.precision_fallback = True
            self.fallback_reason = "NPU precision check failed"
            self.static_session = None
            self.device = "cpu"

    def encode(self, inputs, prompt: str | None = None) -> np.ndarray:
        if any(isinstance(item, dict) for item in inputs):
            if self.mm is None:
                raise RuntimeError(f"multimodal artifacts are not complete in {self.model_dir}")
            return self.mm.encode(inputs, prompt=prompt)
        texts = _with_prompt(list(inputs), prompt)
        enc = self._tokenize(texts, self.max_text_length, pad_to_max=False)
        token_lengths = enc["attention_mask"].sum(axis=1)

        if self.static_session is None:
            return self._run_session(self.safe_session, texts)

        # Short texts stay on NPU. Anything that would be truncated goes to the
        # dynamic CPU FP32 session instead of silently losing content.
        out = np.empty((len(texts), 768), dtype=np.float32)
        for i, (text, length) in enumerate(zip(texts, token_lengths)):
            if self.static_session is not None and length <= self.seq_len:
                try:
                    out[i] = self._run_session(
                        self.static_session, [text], self.seq_len, pad_to_max=True
                    )[0]
                    continue
                except Exception as exc:
                    self.static_session = None
                    self.device = "cpu"
                    self.fallback_reason = f"NPU inference failed: {exc}"
            out[i] = self._run_session(self.safe_session, [text])[0]
        return out


class EmbeddingEngine:
    def __init__(
        self,
        model_root: str | Path = MODEL_ROOT,
        prefer: list[str] | None = None,
        model_dir: str | Path | None = None,
        openvino_variant: str = "w8",
        seq_len: int = DEFAULT_OPENVINO_SEQ_LEN,
        max_text_length: int = MODEL_MAX_SEQUENCE,
        openvino_device: str | None = None,
        onnx_device: str | None = None,
        precision_check: bool = True,
        cache_root: str | Path | None = None,
        verbose: bool = True,
    ):
        self.model_root = Path(model_root)
        self.prefer = prefer or ["onnx", "openvino"]
        self.explicit_dir = Path(model_dir) if model_dir else None
        self.openvino_variant = openvino_variant
        self.seq_len = seq_len
        self.max_text_length = max_text_length
        self.openvino_device = openvino_device
        self.onnx_device = onnx_device
        self.precision_check = precision_check
        self.cache_root = Path(cache_root) if cache_root else None
        self.verbose = verbose
        self.backend = None
        self.error = None
        self.state = EngineState.INIT
        self._run()

    def _transition(self, to: EngineState):
        if to not in _TRANSITIONS[self.state]:
            raise RuntimeError(f"illegal transition: {self.state.name} -> {to.name}")
        if self.verbose:
            print(f"[engine] {self.state.name} -> {to.name}")
        self.state = to

    def _fail(self, err: Exception):
        self.error = err
        self._transition(EngineState.FAILED)
        raise err

    def _run(self):
        try:
            self._transition(EngineState.PROBING)
            candidates = self._probe()
            failures = []
            for index, backend in enumerate(candidates):
                self._transition(EngineState.BACKEND_SELECTED)
                self.backend = backend
                self._transition(EngineState.LOADING)
                try:
                    backend.load()
                    self._transition(EngineState.READY)
                    if self.verbose:
                        extra = (
                            f" ({backend.fallback_reason})"
                            if getattr(backend, "fallback_reason", "")
                            else ""
                        )
                        print(f"[engine] backend={backend.name} device={backend.device}{extra}")
                    return
                except Exception as exc:
                    failures.append(f"{backend.name}: {exc}")
                    self.backend = None
                    if index + 1 < len(candidates):
                        if self.verbose:
                            print(f"[engine] {backend.name} failed, trying next backend: {exc}")
            raise RuntimeError("; ".join(failures))
        except Exception as exc:
            self._fail(exc)

    def _probe(self) -> list:
        if self.explicit_dir:
            if (self.explicit_dir / "embeddinggemma2_text.onnx").exists():
                backend = _OnnxBackend(
                    self.explicit_dir, self.max_text_length, self.onnx_device
                )
            elif any(self.explicit_dir.glob("*.xml")):
                backend = _OpenVinoBackend(
                    self.explicit_dir,
                    self.seq_len,
                    self.max_text_length,
                    self.openvino_device,
                    self.precision_check,
                    self.cache_root,
                )
            else:
                raise RuntimeError(f"no text weights found in {self.explicit_dir}")
            if not backend.available():
                raise RuntimeError(f"{self.explicit_dir}: {backend.status}")
            return [backend]

        if self.openvino_variant == "w8":
            quantized_root = self.model_root / "quantized"
            candidates = sorted(
                set(quantized_root.glob("*__int8__*"))
                | set(quantized_root.glob("*__int8_w8__*"))
            )
            ov_dir = (
                self.model_root / "quantized" / QUANTIZED_DIR_NAME
                if not candidates
                else candidates[0]
            )
        else:
            ov_dir = self.model_root / "openvino"
        registry = {
            "onnx": _OnnxBackend(
                self.model_root / "onnx", self.max_text_length, self.onnx_device
            ),
            "openvino": _OpenVinoBackend(
                ov_dir,
                self.seq_len,
                self.max_text_length,
                self.openvino_device,
                self.precision_check,
                self.cache_root,
            ),
        }
        candidates = []
        for name in self.prefer:
            if name not in registry:
                raise ValueError(f"unknown backend {name!r}; expected onnx or openvino")
            backend = registry[name]
            if backend.available():
                candidates.append(backend)
            elif self.verbose:
                print(f"[engine] backend '{name}' not available: {backend.status}")
        if not candidates:
            raise RuntimeError("no available backend")
        return candidates

    def encode(self, inputs, prompt: str | None = None) -> np.ndarray:
        if self.state is not EngineState.READY:
            raise RuntimeError(f"engine not ready (state={self.state.name})")
        if isinstance(inputs, (str, dict)):
            inputs = [inputs]
        if not isinstance(inputs, (list, tuple)) or not inputs:
            raise ValueError("inputs must be a non-empty str, dict, or list")
        return self.backend.encode(list(inputs), prompt=prompt)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a @ b.T
