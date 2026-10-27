"""HuggingFace model download and cache metadata."""

from __future__ import annotations

import json
import os
import threading
import time
from threading import RLock
from datetime import datetime, timezone
from pathlib import Path

from tqdm.auto import tqdm

from .types import DownloadError, DownloadedModel
from .manifest import write_json_atomic

SOURCES = {
    "huggingface": None,
    "hf": None,
    "hf-mirror": "https://hf-mirror.com",
    "hf_mirror": "https://hf-mirror.com",
}

_DOWNLOAD_LOCK = RLock()

# Xet CAS can reject anonymous mirror downloads with HTTP 401. The normal
# HTTPS resolver is slower but reliable across HuggingFace mirrors.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        item.stat().st_size for item in path.rglob("*") if item.is_file()
    )


def _expected_size(
    repo_name: str, revision: str | None, endpoint: str | None
) -> int | None:
    try:
        from huggingface_hub import HfApi

        info = HfApi(endpoint=endpoint).model_info(
            repo_name, revision=revision, files_metadata=True
        )
        sizes = [sibling.size or 0 for sibling in info.siblings]
        return sum(sizes) if sizes else None
    except Exception:
        return None


def _normalize_source(source: str) -> tuple[str, str | None]:
    key = source.strip().lower()
    if key not in SOURCES:
        raise ValueError(f"source must be one of {sorted(SOURCES)}")
    canonical = "hf-mirror" if key.startswith("hf") and key != "huggingface" and key != "hf" else "huggingface"
    return canonical, SOURCES[key]


def download_model(
    model_dir: str | Path,
    repo_name: str,
    source: str,
    *,
    revision: str | None = None,
    token: str | None = None,
    force: bool = False,
    max_workers: int = 1,
    retries: int = 2,
) -> DownloadedModel:
    if not repo_name or "/" not in repo_name:
        raise ValueError("repo_name must look like 'organization/model'")
    canonical_source, endpoint = _normalize_source(source)
    root = Path(model_dir)
    target = root / repo_name.replace("/", "--")
    metadata_path = target / ".embedding_service_download.json"

    with _DOWNLOAD_LOCK:
        if target.exists() and metadata_path.exists() and not force:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if revision is None or metadata.get("revision") == revision:
                return DownloadedModel(
                    repo_name=repo_name,
                    revision=metadata.get("revision"),
                    source=metadata.get("source", canonical_source),
                    local_path=target,
                )

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise DownloadError("huggingface_hub is required to download models") from exc

        root.mkdir(parents=True, exist_ok=True)
        last_error = None
        for attempt in range(max(1, retries)):
            try:
                expected = _expected_size(repo_name, revision, endpoint)
                initial = _directory_size(target)
                stop = threading.Event()
                progress = tqdm(
                    total=expected,
                    initial=min(initial, expected) if expected else initial,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {repo_name}",
                    leave=True,
                )
                last_size = initial

                def watch():
                    nonlocal last_size
                    while not stop.wait(0.5):
                        current = _directory_size(target)
                        if current > last_size:
                            progress.update(current - last_size)
                            last_size = current

                watcher = threading.Thread(target=watch, daemon=True)
                watcher.start()
                try:
                    snapshot_download(
                        repo_id=repo_name,
                        local_dir=str(target),
                        revision=revision,
                        token=token,
                        endpoint=endpoint,
                        max_workers=max(1, max_workers),
                    )
                finally:
                    stop.set()
                    watcher.join(timeout=2)
                    final_size = _directory_size(target)
                    if final_size > last_size:
                        progress.update(final_size - last_size)
                    progress.close()
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt + 1 < max(1, retries):
                    time.sleep(3 * (attempt + 1))
        if last_error is not None:
            raise DownloadError(
                f"failed to download {repo_name}: {last_error}. "
                "For hf-mirror use --max-workers 1, wait for the rate limit, "
                "or provide HF_TOKEN."
            ) from last_error

        metadata = {
            "repo_name": repo_name,
            "revision": revision,
            "source": canonical_source,
            "endpoint": endpoint,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "local_path": str(target),
        }
        write_json_atomic(metadata_path, metadata)
        return DownloadedModel(
            repo_name=repo_name,
            revision=revision,
            source=canonical_source,
            local_path=target,
        )
