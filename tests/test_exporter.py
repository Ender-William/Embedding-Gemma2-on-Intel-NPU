from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from embedding_service import EmbeddingService
from embedding_service.exporter import (
    validate_multimodal_inputs,
)
from embedding_service.types import ModelArtifactError


class MultimodalPreflightTests(unittest.TestCase):
    def test_missing_component_fails_before_quantization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "vision_encoder.onnx",
                "audio_encoder.onnx",
                "lm_embeds.onnx",
            ):
                (root / name).touch()
            with self.assertRaisesRegex(ModelArtifactError, "token_embeddings"):
                validate_multimodal_inputs(
                    {
                        "vision": str(root / "vision_encoder.onnx"),
                        "audio": str(root / "audio_encoder.onnx"),
                        "lm": str(root / "lm_embeds.onnx"),
                        "token_embeddings": str(root / "token_embeddings.npy"),
                    }
                )

    def test_incomplete_component_set_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "vision_encoder.onnx"
            path.touch()
            with self.assertRaisesRegex(ModelArtifactError, "missing"):
                validate_multimodal_inputs({"vision": str(path)})


class MultimodalAutoExportTests(unittest.TestCase):
    def test_raw_checkpoint_builds_complete_onnx_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "models" / "google--embeddinggemma-2"
            source.mkdir(parents=True)
            (source / "config.json").write_text("{}", encoding="utf-8")
            (source / "model.safetensors").touch()
            asset_dir = root / "models" / "onnx"
            service = EmbeddingService(
                models_root=root / "models",
                cache_root=root / "cache",
                verbose=False,
            )

            def fake_text_export(model_path, output):
                output.mkdir(parents=True, exist_ok=True)
                text_path = output / "embeddinggemma2_text.onnx"
                text_path.touch()
                return text_path

            def fake_multimodal_export(command, **kwargs):
                output = Path(command[command.index("--output-dir") + 1])
                output.mkdir(parents=True, exist_ok=True)
                for filename in (
                    "vision_encoder.onnx",
                    "audio_encoder.onnx",
                    "lm_embeds.onnx",
                    "token_embeddings.npy",
                ):
                    (output / filename).touch()
                return type("Result", (), {"returncode": 0})()

            requested = {
                "vision": str(asset_dir / "vision_encoder.onnx"),
                "audio": str(asset_dir / "audio_encoder.onnx"),
                "lm": str(asset_dir / "lm_embeds.onnx"),
                "token_embeddings": str(asset_dir / "token_embeddings.npy"),
            }
            with (
                patch(
                    "embedding_service.service.export_text_onnx",
                    side_effect=fake_text_export,
                ) as text_export,
                patch(
                    "embedding_service.service.subprocess.run",
                    side_effect=fake_multimodal_export,
                ) as multimodal_export,
            ):
                prepared = service._prepare_multimodal_assets(source, requested)

            self.assertEqual(
                set(prepared),
                {"vision", "audio", "lm", "token_embeddings"},
            )
            self.assertTrue((asset_dir / "embeddinggemma2_text.onnx").is_file())
            text_export.assert_called_once()
            multimodal_export.assert_called_once()


if __name__ == "__main__":
    unittest.main()
