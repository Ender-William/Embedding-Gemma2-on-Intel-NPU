import unittest

import numpy as np
from PIL import Image

from embedding_service.core.multimodal import (
    OnnxMultimodalEncoder,
    _check_embedding,
    _normalize_video,
)


class MultimodalUnitTests(unittest.TestCase):
    def setUp(self):
        self.encoder = object.__new__(OnnxMultimodalEncoder)

    def test_media_placeholders_are_appended(self):
        messages = self.encoder._build_messages(
            {
                "text": "product",
                "image": [
                    Image.new("RGB", (4, 4), (1, 2, 3)),
                    Image.new("RGB", (4, 4), (4, 5, 6)),
                ],
            }
        )
        content = messages[0][0]["content"]
        text = "".join(item.get("text", "") for item in content if item["type"] == "text")
        self.assertEqual(text, "product")
        self.assertEqual(sum(item["type"] == "image" for item in content), 2)

    def test_extra_placeholder_is_rejected(self):
        with self.assertRaises(ValueError):
            self.encoder._build_messages(
                {
                    "text": "<|image|><|image|>",
                    "image": [Image.new("RGB", (4, 4), (1, 2, 3))],
                }
            )

    def test_frame_list_is_one_video(self):
        frames = [Image.new("RGB", (4, 4), (1, 2, 3)) for _ in range(3)]
        videos = _normalize_video(frames)
        self.assertEqual(len(videos), 1)
        self.assertEqual(len(videos[0]), 3)

    def test_non_finite_embedding_is_rejected(self):
        with self.assertRaises(RuntimeError):
            _check_embedding(np.array([[np.nan, 1.0]], dtype=np.float32), "test")


if __name__ == "__main__":
    unittest.main()
