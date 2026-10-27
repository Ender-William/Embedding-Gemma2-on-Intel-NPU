"""High-level EmbeddingService facade."""

from __future__ import annotations

import gc
import atexit
import subprocess
import sys
import time
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np

from .core.cache import clean_cache
from .core.engine import EmbeddingEngine
from .device import (
    check_device_available,
    normalize_device,
    openvino_device_name,
    resolve_backend,
)
from .downloader import download_model
from .exporter import (
    MULTIMODAL_ONNX_FILES,
    export_text_onnx,
    quantize_model,
    validate_multimodal_inputs,
)
from .manifest import read_json
from .types import (
    DownloadedModel,
    EmbeddingResult,
    ModelArtifact,
    ModelArtifactError,
    ServiceState,
    ServiceStateError,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_ROOT = PACKAGE_ROOT / "model"


class EmbeddingService:
    """Stateful EmbeddingGemma 2 service with download/quantize/load/unload."""

    def __init__(
        self,
        models_root: str | Path | None = None,
        cache_root: str | Path = PROJECT_ROOT / "cache",
        default_device: str = "cpu",
        precision_policy: str = "safe",
        strict_device: bool = True,
        verbose: bool = True,
    ):
        self.models_root = Path(models_root or DEFAULT_MODEL_ROOT).expanduser().resolve()
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.default_device = normalize_device(default_device)
        if precision_policy not in ("safe", "performance"):
            raise ValueError("precision_policy must be 'safe' or 'performance'")
        self.precision_policy = precision_policy
        self.strict_device = strict_device
        self.verbose = verbose
        self._lock = RLock()
        self._engine: EmbeddingEngine | None = None
        self._artifact: ModelArtifact | None = None
        self.state = ServiceState.UNLOADED
        self.last_error: Exception | None = None
        self.models_root.mkdir(parents=True, exist_ok=True)
        clean_cache(self.cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        atexit.register(clean_cache, self.cache_root)

    # -------- model acquisition --------

    def _download_model(
        self,
        model_dir: str | Path | None = None,
        repo_name: str | None = None,
        source: str | None = None,
        *,
        revision: str | None = None,
        token: str | None = None,
        force: bool = False,
        max_workers: int = 1,
    ) -> DownloadedModel:
        if not repo_name or not source:
            raise ValueError("repo_name and source are required")
        return download_model(
            model_dir or self.models_root,
            repo_name,
            source,
            revision=revision,
            token=token,
            force=force,
            max_workers=max_workers,
        )

    def _quantization(
        self,
        model_path: str | Path,
        multi_modal: dict[str, str] | None,
        quant: str,
        device: str,
        *,
        backend: str = "auto",
        output_dir: str | Path | None = None,
        calibration_data: str | Path | None = None,
        overwrite: bool = False,
        validate: bool = True,
    ) -> ModelArtifact:
        source = Path(model_path).expanduser().resolve()
        prepared_multi_modal = self._prepare_multimodal_assets(
            source,
            multi_modal,
        )
        if output_dir is None:
            source_name = source.stem if source.is_file() else source.name
            output_dir = (
                self.models_root
                / "quantized"
                / f"{source_name}__{quant.lower()}__{device.lower()}"
            )
        artifact = quantize_model(
            source,
            prepared_multi_modal,
            quant,
            device,
            backend=backend,
            output_dir=output_dir,
            calibration_data=calibration_data,
            overwrite=overwrite,
            cache_root=self.cache_root,
        )
        if validate:
            if not artifact.model_path.exists():
                raise ModelArtifactError(f"missing quantized model: {artifact.model_path}")
            component_paths = {
                *artifact.components.values(),
                *artifact.multi_modal.values(),
            }
            for component_path in component_paths:
                if not Path(component_path).exists():
                    raise ModelArtifactError(f"missing component: {component_path}")
        else:
            artifact.validated = False
        clean_cache(self.cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        return artifact

    def _prepare_multimodal_assets(
        self,
        source: Path,
        multi_modal: dict[str, str] | None,
    ) -> dict[str, str] | None:
        """Export missing ONNX components from a downloaded HF checkpoint."""
        if not multi_modal:
            return None

        provided = {
            key: Path(value).expanduser().resolve()
            for key, value in multi_modal.items()
        }
        unknown = sorted(set(provided) - set(MULTIMODAL_ONNX_FILES))
        if unknown:
            raise ModelArtifactError(f"unknown multimodal components: {unknown}")

        parents = {path.parent for path in provided.values()}
        asset_dir = (
            next(iter(parents))
            if len(parents) == 1
            else self.models_root / "onnx"
        )
        defaults = {
            key: asset_dir / filename
            for key, filename in MULTIMODAL_ONNX_FILES.items()
        }
        all_paths = {
            key: provided.get(key, defaults[key])
            for key in MULTIMODAL_ONNX_FILES
        }
        missing = {
            key: path for key, path in all_paths.items() if not path.is_file()
        }
        text_onnx = asset_dir / "embeddinggemma2_text.onnx"
        if not missing and text_onnx.is_file():
            return {key: str(path) for key, path in all_paths.items()}

        is_checkpoint = (
            source.is_dir()
            and (source / "config.json").is_file()
            and (
                (source / "model.safetensors").is_file()
                or (source / "pytorch_model.bin").is_file()
            )
        )
        if not is_checkpoint:
            validate_multimodal_inputs(
                {key: str(path) for key, path in all_paths.items()}
            )
            return {key: str(path) for key, path in all_paths.items()}

        exporter = PROJECT_ROOT / "tools" / "export_multimodal_onnx.py"
        if not exporter.is_file():
            raise ModelArtifactError(
                f"multimodal exporter is missing: {exporter}"
            )
        if self.verbose:
            print(
                "[prepare] multimodal ONNX artifacts are missing; exporting "
                f"from {source} to {asset_dir}"
            )
        if not text_onnx.is_file():
            if self.verbose:
                print(f"[prepare] exporting text ONNX to {text_onnx}")
            export_text_onnx(source, asset_dir)
        if not missing:
            return {key: str(path) for key, path in all_paths.items()}

        result = subprocess.run(
            [
                sys.executable,
                str(exporter),
                "--model-dir",
                str(source),
                "--output-dir",
                str(asset_dir),
            ],
            cwd=str(PROJECT_ROOT),
            check=False,
        )
        if result.returncode != 0:
            raise ModelArtifactError(
                "automatic multimodal ONNX export failed with exit code "
                f"{result.returncode}"
            )
        exported = {
            key: provided[key] if provided.get(key, defaults[key]).is_file()
            else defaults[key]
            for key in MULTIMODAL_ONNX_FILES
        }
        validate_multimodal_inputs(
            {key: str(path) for key, path in exported.items()}
        )
        return {key: str(path) for key, path in exported.items()}

    # -------- lifecycle --------

    def _resolve_artifact(self, model_info: str | Path | dict | ModelArtifact) -> ModelArtifact:
        if isinstance(model_info, ModelArtifact):
            return model_info
        if isinstance(model_info, dict):
            return ModelArtifact.from_dict(model_info)
        path = Path(model_info).expanduser().resolve()
        if path.is_file() and path.name == "model_info.json":
            return ModelArtifact.from_dict(read_json(path))
        if path.is_dir():
            manifest = path / "model_info.json"
            if manifest.exists():
                return ModelArtifact.from_dict(read_json(manifest))
            if any(path.glob("*.xml")):
                return ModelArtifact(
                    model_path=path,
                    source_model_path=None,
                    backend="openvino",
                    format="openvino_ir",
                    quant="UNKNOWN",
                    device=self.default_device,
                    components={},
                )
            if any(path.glob("*.onnx")):
                return ModelArtifact(
                    model_path=path,
                    source_model_path=None,
                    backend="onnxruntime",
                    format="onnx",
                    quant="UNKNOWN",
                    device=self.default_device,
                    components={},
                )
        raise ModelArtifactError(f"no model manifest or weights found at {path}")

    def _load(
        self,
        model_info: str | Path | dict | ModelArtifact,
        *,
        device: str | None = None,
        allow_fallback: bool = False,
        warmup: bool = True,
    ) -> None:
        with self._lock:
            artifact = self._resolve_artifact(model_info)
            target_device = normalize_device(device or artifact.device or self.default_device)
            backend = artifact.backend
            if backend not in ("onnxruntime", "openvino"):
                raise ModelArtifactError(f"unsupported backend {backend!r}")
            if backend == "onnxruntime" and target_device not in ("cpu", "cuda"):
                raise ModelArtifactError("onnxruntime artifacts support cpu or cuda")
            if backend == "openvino" and target_device == "cuda":
                raise ModelArtifactError("OpenVINO artifacts cannot target cuda")
            check_device_available(target_device)

            if self._engine is not None:
                self._off_load()

            self.state = ServiceState.LOADING
            self.last_error = None
            try:
                openvino_device = (
                    openvino_device_name(target_device)
                    if backend == "openvino"
                    else None
                )
                engine = EmbeddingEngine(
                    model_dir=artifact.model_path,
                    prefer=[backend],
                    openvino_device=openvino_device,
                    onnx_device=target_device if backend == "onnxruntime" else None,
                    precision_check=self.precision_policy == "safe",
                    cache_root=self.cache_root,
                    verbose=self.verbose,
                )
                actual = (engine.backend.device or "").lower()
                expected = {
                    "cuda": "cuda",
                    "intel_npu": "npu",
                    "intel_gpu": "gpu",
                    "cpu": "cpu",
                }[target_device]
                if self.strict_device and not actual.startswith(expected):
                    raise ModelArtifactError(
                        f"requested {target_device}, runtime selected {actual or 'unknown'}"
                    )
                self._engine = engine
                self._artifact = artifact
                if warmup:
                    self._engine.encode(["warmup"], prompt="query")
                self.state = ServiceState.READY
            except Exception as exc:
                self._engine = None
                self._artifact = None
                self.state = ServiceState.FAILED
                self.last_error = exc
                clean_cache(self.cache_root)
                raise
            finally:
                clean_cache(self.cache_root)

    def _off_load(self) -> bool:
        with self._lock:
            if self.state is ServiceState.UNLOADED and self._engine is None:
                clean_cache(self.cache_root)
                return False
            self.state = ServiceState.UNLOADING
            self._engine = None
            self._artifact = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            clean_cache(self.cache_root)
            self.state = ServiceState.UNLOADED
            return True

    # -------- inference --------

    def run(
        self,
        inputs: Any,
        *,
        prompt: str | None = None,
        normalize: bool = True,
        return_metadata: bool = True,
    ) -> EmbeddingResult:
        with self._lock:
            if self.state is not ServiceState.READY or self._engine is None:
                raise ServiceStateError(f"service is not ready: {self.state.value}")
            if isinstance(inputs, (str, dict)):
                items = [inputs]
            elif isinstance(inputs, (list, tuple)) and inputs:
                items = list(inputs)
            else:
                raise ValueError("inputs must be a non-empty str, dict, or list")
            started = time.perf_counter()
            embeddings = np.asarray(self._engine.encode(items, prompt=prompt), dtype=np.float32)
            elapsed = (time.perf_counter() - started) * 1000
            if normalize:
                norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
                embeddings = embeddings / np.maximum(norms, 1e-9)
            artifact = self._artifact
            if artifact is None:
                raise ServiceStateError("artifact metadata is missing")
            return EmbeddingResult(
                embeddings=embeddings,
                dimension=int(embeddings.shape[-1]),
                model_path=str(artifact.model_path),
                backend=artifact.backend,
                device=artifact.device,
                quant=artifact.quant,
                elapsed_ms=elapsed,
                prompt=prompt,
                inputs=items,
            )

    def warmup(self) -> EmbeddingResult:
        return self.run(["warmup"], prompt="query")

    def similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.asarray(a, dtype=np.float32) @ np.asarray(b, dtype=np.float32).T

    def search(self, query, candidates: list, top_k: int = 5):
        q = self.run(query, prompt="query").embeddings
        docs = self.run(candidates, prompt="document").embeddings
        scores = (docs @ q[0]).reshape(-1)
        order = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i]), candidates[i]) for i in order]

    # -------- inspection and aliases --------

    def status(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "model_path": str(self._artifact.model_path) if self._artifact else None,
            "backend": self._artifact.backend if self._artifact else None,
            "device": self._artifact.device if self._artifact else None,
            "quant": self._artifact.quant if self._artifact else None,
            "last_error": str(self.last_error) if self.last_error else None,
        }

    def get_model_info(self, model_path: str | Path) -> ModelArtifact:
        path = Path(model_path)
        manifest = path / "model_info.json"
        if not manifest.exists():
            raise ModelArtifactError(f"missing model_info.json: {manifest}")
        return ModelArtifact.from_dict(read_json(manifest))

    def list_models(self) -> list[ModelArtifact]:
        found = []
        for manifest in sorted(self.models_root.rglob("model_info.json")):
            try:
                found.append(ModelArtifact.from_dict(read_json(manifest)))
            except Exception:
                continue
        return found

    download_model = _download_model
    quantize = _quantization
    load = _load
    unload = _off_load

    def __enter__(self) -> "EmbeddingService":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._off_load()
