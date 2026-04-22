from __future__ import annotations

import sys
from pathlib import Path


DEFAULT_TORCH_SITE_PACKAGES = (
    Path(r"D:\Kaggle\JaneStreet\2025_v2\.venv\Lib\site-packages"),
)


def ensure_torch():
    try:
        import torch  # type: ignore

        return torch
    except ModuleNotFoundError:
        for path in DEFAULT_TORCH_SITE_PACKAGES:
            if path.exists():
                sys.path.insert(0, str(path))

        import torch  # type: ignore

        return torch

