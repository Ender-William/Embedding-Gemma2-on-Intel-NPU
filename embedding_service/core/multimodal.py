"""Pure ONNX / OpenVINO multimodal embedding pipelines.

Inputs:
    str: text
    dict: {"text": "...", "image": ..., "video": ..., "audio": ...}

Media values may be a single path or a list of paths. A list of PIL/numpy
frames is also accepted as one video. Video files are decoded with PyAV at
one frame per second by default, so torchcodec and system FFmpeg DLLs are not
required.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import ml_dtypes  # noqa: F401 - registers bfloat16 for np.load
import numpy as np

from . import runtime_env  # noqa: F401 - register bundled NVIDIA DLLs before ORT
from .cache import temporary_compile_cache
import onnxruntime as ort
from PIL import Image
from transformers import AutoProcessor
from transformers.video_utils import VideoMetadata

try:
    import av
except ImportError:  # pragma: no cover - reported when video is actually used
    av = None


IMAGE_TOKEN_ID = 258880
AUDIO_TOKEN_ID = 258881
VIDEO_TOKEN_ID = 258884
PAD_TOKEN_ID = 0

AUDIO_WINDOW_FRAMES = 1000  # ~10 seconds at the model's 100 feature frames/s
VIDEO_FPS = 1.0
VIDEO_MAX_FRAMES = 58
VISION_CANVAS = 2520
MODEL_MAX_SEQUENCE = 8192

PROMPTS = {
    "query": "task: search result | query: ",
    "document": "title: none | text: ",
}

PLACEHOLDER = {"image": "<|image|>", "video": "<|video|>", "audio": "<|audio|>"}


def _is_path(value: Any) -> bool:
    return isinstance(value, (str, os.PathLike))


def _is_frame(value: Any) -> bool:
    return isinstance(value, (Image.Image, np.ndarray))


def decode_video_frames(
    path: str | os.PathLike,
    fps: float = VIDEO_FPS,
    max_frames: int = VIDEO_MAX_FRAMES,
) -> list[Image.Image]:
    """Decode an MP4/MKV/etc. into PIL frames using PyAV.

    PyAV wheels bundle their own FFmpeg libraries on Windows, avoiding the
    torchcodec/system-FFmpeg DLL problem.
    """
    if av is None:
        raise RuntimeError(
            "video input requires PyAV; install it with `uv sync --extra export` "
            "or `uv pip install av`"
        )
    if fps <= 0:
        raise ValueError("fps must be positive")

    frames: list[Image.Image] = []
    container = av.open(str(path))
    try:
        if not container.streams.video:
            raise ValueError(f"no video stream in {path}")
        stream = container.streams.video[0]
        step = 1.0 / fps
        next_time = 0.0
        for frame in container.decode(stream):
            # frame.time is the presentation timestamp in seconds.
            timestamp = float(frame.time or 0.0)
            if timestamp + 1e-6 < next_time:
                continue
            frames.append(frame.to_image())
            next_time = timestamp + step
            if len(frames) >= max_frames:
                break
    finally:
        container.close()

    if not frames:
        raise ValueError(f"could not decode any frames from {path}")
    return frames


def _normalize_video(value: Any) -> list[list[Image.Image]]:
    """Return a list of videos, each represented as a frame list."""
    if _is_path(value):
        return [decode_video_frames(value)]
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("video list is empty")
        if all(_is_frame(item) for item in value):
            return [[item if isinstance(item, Image.Image) else Image.fromarray(item) for item in value]]
        if all(_is_path(item) for item in value):
            return [decode_video_frames(item) for item in value]
        if all(isinstance(item, (list, tuple)) for item in value):
            return [
                [
                    frame if isinstance(frame, Image.Image) else Image.fromarray(frame)
                    for frame in item
                ]
                for item in value
            ]
    raise TypeError(
        "video must be a path, a list of paths, a frame list, or a list of frame lists"
    )


def _normalize_media(value: Any, kind: str) -> list[Any]:
    if kind == "video":
        return _normalize_video(value)
    if kind == "image":
        items = list(value) if isinstance(value, (list, tuple)) else [value]
        images = []
        for item in items:
            if isinstance(item, Image.Image):
                images.append(item.convert("RGB"))
            elif isinstance(item, np.ndarray):
                images.append(Image.fromarray(item).convert("RGB"))
            elif _is_path(item):
                with Image.open(item) as image:
                    images.append(image.convert("RGB"))
            else:
                raise TypeError("image must be a path, PIL image, numpy array, or list")
        return images
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"{kind} list is empty")
        return list(value)
    return [value]


def _check_embedding(embeds: np.ndarray, name: str) -> np.ndarray:
    embeds = np.asarray(embeds, dtype=np.float32)
    if not np.isfinite(embeds).all():
        raise RuntimeError(f"{name} produced non-finite values; refusing degraded embedding")
    norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    if np.any(norms <= 1e-8):
        raise RuntimeError(f"{name} produced a zero embedding")
    return embeds / norms


class OnnxMultimodalEncoder:
    def __init__(self, model_dir: str | Path, providers=None):
        self.model_dir = Path(model_dir)
        providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess_vision = ort.InferenceSession(
            str(self.model_dir / "vision_encoder.onnx"), providers=providers
        )
        self.sess_audio = ort.InferenceSession(
            str(self.model_dir / "audio_encoder.onnx"), providers=providers
        )
        self.sess_lm = ort.InferenceSession(
            str(self.model_dir / "lm_embeds.onnx"), providers=providers
        )
        self._init_common(self.model_dir)

    def _init_common(self, model_dir: Path):
        self.processor = AutoProcessor.from_pretrained(str(model_dir))
        raw = np.load(model_dir / "token_embeddings.npy")
        self.table = raw.view(ml_dtypes.bfloat16).astype(np.float32)
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        self.embed_scale = math.sqrt(cfg["text_config"]["hidden_size"])

    def _build_messages(self, inp: dict) -> list:
        text = inp.get("text")
        media = {
            kind: _normalize_media(inp[kind], kind)
            for kind in ("image", "video", "audio")
            if kind in inp
        }
        if text is None and not media:
            raise ValueError("multimodal input must contain text or at least one media item")

        if text is None:
            text = "".join(
                PLACEHOLDER[kind] * len(items) for kind, items in media.items()
            )
        else:
            # Append missing placeholders; reject placeholders without media.
            for kind, items in media.items():
                ph = PLACEHOLDER[kind]
                count = text.count(ph)
                if count > len(items):
                    raise ValueError(
                        f"{kind} placeholders={count}, media={len(items)}; "
                        "every placeholder needs one media item"
                    )
                if count < len(items):
                    text += ph * (len(items) - count)

        queues = {PLACEHOLDER[kind]: (kind, list(items)) for kind, items in media.items()}
        content = []
        remaining = text
        while remaining:
            hits = [
                (remaining.find(ph), ph)
                for ph in queues
                if remaining.find(ph) >= 0
            ]
            if not hits:
                content.append({"type": "text", "text": remaining})
                break
            pos, ph = min(hits)
            if pos > 0:
                content.append({"type": "text", "text": remaining[:pos]})
            kind, items = queues[ph]
            if not items:
                raise ValueError(f"no {kind} item left for placeholder {ph}")
            content.append({"type": kind, kind: items.pop(0)})
            remaining = remaining[pos + len(ph):]
        return [[{"role": "user", "content": content}]]

    def _preprocess(self, inp: str | dict, prompt: str | None) -> dict:
        if prompt is not None and prompt not in PROMPTS:
            raise ValueError(f"unknown prompt {prompt!r}; expected one of {sorted(PROMPTS)}")
        if isinstance(inp, str):
            text = (PROMPTS[prompt] if prompt else "") + inp
            msgs = [[{"role": "user", "content": [{"type": "text", "text": text}]}]]
            videos_kwargs = None
        elif isinstance(inp, dict):
            inp = dict(inp)
            if prompt and inp.get("text"):
                inp["text"] = PROMPTS[prompt] + str(inp["text"])
            videos_kwargs = None
            if "video" in inp:
                videos = _normalize_video(inp["video"])
                inp["video"] = videos
                videos_kwargs = {
                    "video_metadata": [
                        VideoMetadata(
                            total_num_frames=len(frames),
                            fps=VIDEO_FPS,
                            width=frames[0].width,
                            height=frames[0].height,
                            duration=len(frames) / VIDEO_FPS,
                            frames_indices=list(range(len(frames))),
                        )
                        for frames in videos
                    ]
                }
            msgs = self._build_messages(inp)
        else:
            raise TypeError("input must be str or dict")
        kwargs = {}
        if videos_kwargs is not None:
            kwargs["processor_kwargs"] = {"videos_kwargs": videos_kwargs}
        return self.processor.apply_chat_template(
            msgs,
            tokenize=True,
            return_dict=True,
            return_tensors="np",
            add_generation_prompt=False,
            **kwargs,
        )

    def _encode_image(self, pixel_values, position_ids) -> np.ndarray:
        (soft,) = self.sess_vision.run(
            ["image_embeds"],
            {"pixel_values": pixel_values, "image_position_ids": position_ids},
        )
        return soft

    def _encode_audio(self, features, mask) -> np.ndarray:
        chunks = []
        for start in range(0, features.shape[1], AUDIO_WINDOW_FRAMES):
            f = features[:, start:start + AUDIO_WINDOW_FRAMES]
            m = mask[:, start:start + AUDIO_WINDOW_FRAMES]
            pad = AUDIO_WINDOW_FRAMES - f.shape[1]
            if pad:
                f = np.pad(f, ((0, 0), (0, pad), (0, 0)))
                m = np.pad(m, ((0, 0), (0, pad)))
            soft, valid = self.sess_audio.run(
                ["audio_embeds", "audio_valid_mask"],
                {
                    "input_features": f.astype(np.float32),
                    "input_features_mask": m.astype(np.float32),
                },
            )
            chunks.append(soft[0, : int(valid.astype(bool).sum())])
        return np.concatenate(chunks, axis=0)

    def _merge_embeds(self, feats: dict) -> tuple[np.ndarray, np.ndarray]:
        ids = feats["input_ids"][0]
        mask = feats["attention_mask"][0].astype(np.float32)
        mm = (ids == IMAGE_TOKEN_ID) | (ids == AUDIO_TOKEN_ID) | (ids == VIDEO_TOKEN_ID)
        embeds = self.table[np.where(mm, PAD_TOKEN_ID, ids)] * self.embed_scale

        def inject(kind: str, token_id: int, soft: np.ndarray):
            pos = np.where(ids == token_id)[0]
            if len(pos) != len(soft):
                raise ValueError(
                    f"{kind} soft tokens={len(soft)}, placeholders={len(pos)}"
                )
            embeds[pos] = soft

        if "pixel_values" in feats:
            image_rows = feats["pixel_values"]
            soft = np.concatenate(
                [
                    self._encode_image(image_rows[i:i + 1], feats["image_position_ids"][i:i + 1])
                    for i in range(image_rows.shape[0])
                ],
                axis=0,
            )
            inject("image", IMAGE_TOKEN_ID, soft)

        if "pixel_values_videos" in feats:
            video_rows = feats["pixel_values_videos"]
            soft = np.concatenate(
                [
                    self._encode_image(
                        video_rows[i:i + 1], feats["video_position_ids"][i:i + 1]
                    )
                    for i in range(video_rows.shape[0])
                ],
                axis=0,
            )
            inject("video", VIDEO_TOKEN_ID, soft)

        if "input_features" in feats:
            fmask = feats["input_features_mask"].astype(bool)
            inject("audio", AUDIO_TOKEN_ID, self._encode_audio(feats["input_features"], fmask))

        return embeds[None], mask[None]

    def _lm_hidden(self, embeds: np.ndarray, mask: np.ndarray) -> np.ndarray:
        (hidden,) = self.sess_lm.run(
            ["last_hidden_state"],
            {
                "inputs_embeds": embeds.astype(np.float32),
                "attention_mask": mask.astype(np.int64),
            },
        )
        return hidden

    def _run_lm(self, embeds: np.ndarray, mask: np.ndarray) -> np.ndarray:
        hidden = self._lm_hidden(embeds, mask)
        mask = mask[:, : hidden.shape[1]]
        pooled = (hidden * mask[..., None]).sum(1) / np.maximum(
            mask.sum(1, keepdims=True), 1e-9
        )
        return _check_embedding(pooled, type(self).__name__)[0]

    def encode(self, inputs, prompt: str | None = None) -> np.ndarray:
        single = isinstance(inputs, (str, dict))
        if single:
            inputs = [inputs]
        embs = [
            self._run_lm(*self._merge_embeds(self._preprocess(inp, prompt)))
            for inp in inputs
        ]
        embs = np.stack(embs)
        return embs[0] if single else embs

    def similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a @ b.T


class OpenVinoMultimodalEncoder(OnnxMultimodalEncoder):
    """OpenVINO multimodal encoder with a safe fallback for long sequences.

    NPU accepts FP16 only on this hardware. CPU fallback uses FP32 activations.
    When the merged sequence exceeds the static NPU length, it is run on CPU
    with the dynamic LM IR instead of being silently truncated.
    """

    def __init__(
        self,
        model_dir: str | Path,
        device="NPU",
        seq_len: int = 512,
        cache_root: str | Path | None = None,
    ):
        import openvino as ov

        self.model_dir = Path(model_dir)
        self.seq_len = seq_len
        self.cache_root = cache_root
        self.core = ov.Core()
        if isinstance(device, str):
            if device.upper() == "NPU":
                # The vision tower hangs some Intel NPU drivers; keep it on CPU.
                device = {"vision": "CPU", "audio": "NPU", "lm": "NPU"}
            else:
                device = {"vision": device, "audio": device, "lm": device}
        self.device = dict(device)
        self._dynamic_lm = None

        def cfg(dev: str) -> dict:
            return {"INFERENCE_PRECISION_HINT": "f16" if dev.upper() == "NPU" else "f32"}

        def load(name, dev, reshape=None):
            m = self.core.read_model(str(self.model_dir / name))
            if reshape:
                m.reshape(reshape)
            with temporary_compile_cache(self.cache_root) as cache_dir:
                config = cfg(dev)
                if cache_dir is not None:
                    config["CACHE_DIR"] = cache_dir
                return self.core.compile_model(m, dev, config)

        self.ov_vision = load("vision_encoder.xml", self.device["vision"])
        self.ov_audio = load("audio_encoder.xml", self.device["audio"])
        self.ov_lm = load(
            "lm_embeds.xml",
            self.device["lm"],
            {"inputs_embeds": [1, seq_len, 512], "attention_mask": [1, seq_len]},
        )
        self._init_common(self.model_dir)

    def _encode_image(self, pixel_values, position_ids) -> np.ndarray:
        n_valid = int((position_ids != -1).all(axis=-1).sum()) // 9
        P = pixel_values.shape[1]
        if P > VISION_CANVAS:
            raise ValueError(f"vision canvas has {P} patches, max supported is {VISION_CANVAS}")
        if P < VISION_CANVAS:
            pixel_values = np.pad(
                pixel_values, ((0, 0), (0, VISION_CANVAS - P), (0, 0))
            )
            position_ids = np.pad(
                position_ids,
                ((0, 0), (0, VISION_CANVAS - P), (0, 0)),
                constant_values=-1,
            )
        result = self.ov_vision(
            {
                "pixel_values": pixel_values.astype(np.float32),
                "image_position_ids": position_ids.astype(np.int64),
            }
        )
        soft = np.asarray(result[self.ov_vision.output(0)], dtype=np.float32)
        return soft[:n_valid]

    def _encode_audio(self, features, mask) -> np.ndarray:
        chunks = []
        for start in range(0, features.shape[1], AUDIO_WINDOW_FRAMES):
            f = features[:, start:start + AUDIO_WINDOW_FRAMES]
            m = mask[:, start:start + AUDIO_WINDOW_FRAMES]
            pad = AUDIO_WINDOW_FRAMES - f.shape[1]
            if pad:
                f = np.pad(f, ((0, 0), (0, pad), (0, 0)))
                m = np.pad(m, ((0, 0), (0, pad)))
            result = self.ov_audio(
                {
                    "input_features": f.astype(np.float32),
                    "input_features_mask": m.astype(np.float32),
                }
            )
            soft = np.asarray(result[self.ov_audio.output(0)], dtype=np.float32)[0]
            valid = np.asarray(result[self.ov_audio.output(1)]).astype(bool)[0]
            chunks.append(soft[: int(valid.sum())])
        return np.concatenate(chunks, axis=0)

    def _dynamic_cpu_lm(self):
        if self._dynamic_lm is None:
            m = self.core.read_model(str(self.model_dir / "lm_embeds.xml"))
            with temporary_compile_cache(self.cache_root) as cache_dir:
                config = {"INFERENCE_PRECISION_HINT": "f32"}
                if cache_dir is not None:
                    config["CACHE_DIR"] = cache_dir
                self._dynamic_lm = self.core.compile_model(m, "CPU", config)
        return self._dynamic_lm

    def _lm_hidden(self, embeds: np.ndarray, mask: np.ndarray) -> np.ndarray:
        L = embeds.shape[1]
        if L > MODEL_MAX_SEQUENCE:
            raise ValueError(
                f"multimodal sequence has {L} tokens, official limit is {MODEL_MAX_SEQUENCE}"
            )
        if L > self.seq_len:
            compiled = self._dynamic_cpu_lm()
            result = compiled(
                {
                    "inputs_embeds": embeds.astype(np.float32),
                    "attention_mask": mask.astype(np.int64),
                }
            )
            output = compiled.output(0)
        else:
            pad = self.seq_len - L
            result = self.ov_lm(
                {
                    "inputs_embeds": np.pad(
                        embeds, ((0, 0), (0, pad), (0, 0))
                    ).astype(np.float32),
                    "attention_mask": np.pad(mask, ((0, 0), (0, pad))).astype(np.int64),
                }
            )
            output = self.ov_lm.output(0)
        hidden = np.asarray(result[output], dtype=np.float32)
        return hidden[:, :L]
