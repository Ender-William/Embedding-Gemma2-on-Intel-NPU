"""Temporary compile-cache helpers."""

from __future__ import annotations

import gc
import os
import shutil
import stat
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def temporary_compile_cache(root: str | Path | None):
    """Create a unique cache directory and remove it after compilation."""
    if root is None:
        yield None
        return
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    target = base / f"compile-{uuid.uuid4().hex}"
    target.mkdir()
    try:
        yield str(target)
    finally:
        shutil.rmtree(target, ignore_errors=True)


def clean_cache(root: str | Path | None) -> None:
    if root is None:
        return
    target = Path(root)
    for _ in range(10):
        if not target.exists():
            return
        gc.collect()
        try:
            shutil.rmtree(target, onerror=_remove_readonly)
            return
        except OSError:
            time.sleep(0.2)


def _remove_readonly(function, path, _exc_info):
    os.chmod(path, stat.S_IWRITE)
    function(path)
