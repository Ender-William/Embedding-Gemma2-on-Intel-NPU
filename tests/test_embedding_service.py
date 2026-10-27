from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from embedding_service import (
    DEFAULT_MODEL_ROOT,
    EmbeddingService,
    ServiceState,
    ServiceStateError,
)
from embedding_service.downloader import download_model

ROOT = Path(__file__).resolve().parent.parent


class DownloaderTests(unittest.TestCase):
    def test_hf_mirror_download_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("embedding_service.downloader._expected_size", return_value=None),
                patch("huggingface_hub.snapshot_download") as mocked,
            ):
                mocked.side_effect = lambda **kwargs: Path(kwargs["local_dir"]).mkdir(
                    parents=True, exist_ok=True
                )
                result = download_model(
                    directory, "org/model", "hf-mirror", revision="main"
                )
            self.assertTrue((result.local_path / ".embedding_service_download.json").exists())
            self.assertEqual(result.source, "hf-mirror")
            self.assertEqual(mocked.call_args.kwargs["max_workers"], 1)
            self.assertEqual(
                mocked.call_args.kwargs["endpoint"], "https://hf-mirror.com"
            )


class ServiceLifecycleTests(unittest.TestCase):
    def test_default_model_root_is_package_model_dir(self):
        service = EmbeddingService(cache_root=Path(tempfile.gettempdir()) / "embed-cache-test")
        self.assertEqual(service.models_root, DEFAULT_MODEL_ROOT)
        service.unload()

    def test_service_download_uses_models_root_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "model"
            service = EmbeddingService(models_root=root, cache_root=Path(directory) / "cache")
            with (
                patch("embedding_service.downloader._expected_size", return_value=None),
                patch("huggingface_hub.snapshot_download") as mocked,
            ):
                mocked.side_effect = lambda **kwargs: Path(kwargs["local_dir"]).mkdir(
                    parents=True, exist_ok=True
                )
                result = service._download_model(
                    repo_name="org/model", source="huggingface"
                )
            self.assertTrue(result.local_path.is_relative_to(root))
            service.unload()

    def test_load_run_unload_and_cache_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            service = EmbeddingService(
                models_root=Path(directory) / "models",
                cache_root=cache,
                default_device="cpu",
                verbose=False,
            )
            try:
                service.load(
                    ROOT / "embedding_service" / "model" / "onnx",
                    device="cpu",
                    warmup=False,
                )
                self.assertEqual(service.state, ServiceState.READY)
                self.assertFalse(cache.exists())
                result = service.run(["hello"], prompt="query")
                self.assertEqual(result.embeddings.shape, (1, 768))
                self.assertTrue(service.unload())
                self.assertEqual(service.state, ServiceState.UNLOADED)
                with self.assertRaises(ServiceStateError):
                    service.run(["hello"])
            finally:
                service.unload()

    def test_int8_quantization_creates_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            service = EmbeddingService(
                models_root=Path(directory) / "models",
                cache_root=Path(directory) / "cache",
                default_device="intel_npu",
                verbose=False,
            )
            output = Path(directory) / "quantized"
            artifact = service.quantize(
                ROOT
                / "embedding_service"
                / "model"
                / "onnx"
                / "embeddinggemma2_text.onnx",
                None,
                "INT8",
                "intel_npu",
                output_dir=output,
            )
            self.assertEqual(artifact.quant, "INT8_W8")
            self.assertTrue((output / "model_info.json").exists())
            self.assertTrue(Path(artifact.components["text"]).exists())


if __name__ == "__main__":
    unittest.main()
