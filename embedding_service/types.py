"""Public types and exceptions for EmbeddingService."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class ServiceState(Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    UNLOADING = "unloading"
    FAILED = "failed"


class EmbeddingServiceError(RuntimeError):
    pass


class ServiceStateError(EmbeddingServiceError):
    pass


class ModelArtifactError(EmbeddingServiceError):
    pass


class DeviceUnavailableError(EmbeddingServiceError):
    pass


class UnsupportedQuantizationError(EmbeddingServiceError):
    pass


class DownloadError(EmbeddingServiceError):
    pass


@dataclass
class DownloadedModel:
    repo_name: str
    revision: str | None
    source: str
    local_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_name": self.repo_name,
            "revision": self.revision,
            "source": self.source,
            "local_path": str(self.local_path),
        }


@dataclass
class ModelArtifact:
    model_path: Path
    source_model_path: Path | None
    backend: str
    format: str
    quant: str
    device: str
    components: dict[str, str] = field(default_factory=dict)
    multi_modal: dict[str, str] = field(default_factory=dict)
    precision_policy: str = ""
    manifest: Path | None = None
    validated: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["model_path"] = str(self.model_path)
        data["source_model_path"] = (
            str(self.source_model_path) if self.source_model_path else None
        )
        data["manifest"] = str(self.manifest) if self.manifest else None
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelArtifact":
        return cls(
            model_path=Path(data["model_path"]),
            source_model_path=(
                Path(data["source_model_path"]) if data.get("source_model_path") else None
            ),
            backend=data["backend"],
            format=data["format"],
            quant=data["quant"],
            device=data["device"],
            components=dict(data.get("components", {})),
            multi_modal=dict(data.get("multi_modal", {})),
            precision_policy=data.get("precision_policy", ""),
            manifest=Path(data["manifest"]) if data.get("manifest") else None,
            validated=bool(data.get("validated", False)),
        )


@dataclass
class EmbeddingResult:
    embeddings: np.ndarray
    dimension: int
    model_path: str
    backend: str
    device: str
    quant: str
    elapsed_ms: float
    prompt: str | None
    inputs: list[Any]

    def to_dict(self, include_embeddings: bool = False) -> dict[str, Any]:
        data = {
            "dimension": self.dimension,
            "model_path": self.model_path,
            "backend": self.backend,
            "device": self.device,
            "quant": self.quant,
            "elapsed_ms": self.elapsed_ms,
            "prompt": self.prompt,
            "inputs": self.inputs,
        }
        if include_embeddings:
            data["embeddings"] = self.embeddings.tolist()
        return data
