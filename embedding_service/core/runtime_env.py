"""Register bundled NVIDIA runtime DLL directories on Windows.

onnxruntime-gpu depends on CUDA/cuDNN DLLs, but the nvidia-* wheels place
them under site-packages/nvidia/... instead of a directory on PATH.
"""

from __future__ import annotations

import os
import sysconfig
from pathlib import Path

_DLL_HANDLES = []


def configure_nvidia_dlls() -> list[Path]:
    if os.name != "nt":
        return []
    site_packages = Path(sysconfig.get_paths()["purelib"])
    relative = (
        Path("torch") / "lib",
        Path("nvidia") / "cu13" / "bin" / "x86_64",
        Path("nvidia") / "cudnn" / "bin",
    )
    added = []
    for rel in relative:
        directory = site_packages / rel
        if not directory.exists():
            continue
        os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
        _DLL_HANDLES.append(os.add_dll_directory(str(directory)))
        added.append(directory)
    return added


configure_nvidia_dlls()
