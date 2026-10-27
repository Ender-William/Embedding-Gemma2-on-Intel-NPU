"""EmbeddingGemma 2 service package."""

from .service import DEFAULT_MODEL_ROOT, EmbeddingService
from .types import (
    EmbeddingResult,
    ModelArtifact,
    ServiceState,
    ServiceStateError,
    UnsupportedQuantizationError,
)

__all__ = [
    "EmbeddingService",
    "DEFAULT_MODEL_ROOT",
    "EmbeddingResult",
    "ModelArtifact",
    "ServiceState",
    "ServiceStateError",
    "UnsupportedQuantizationError",
]
