from __future__ import annotations

from typing import Mapping

from .utils.externals import ensure_torch

torch = ensure_torch()


def build_optimizer(config: Mapping, params):
    cfg = dict(config)
    name = str(cfg.pop("name", "adamw")).lower()
    if name == "adamw":
        return torch.optim.AdamW(params, **cfg)
    if name == "adam":
        return torch.optim.Adam(params, **cfg)
    if name == "sgd":
        return torch.optim.SGD(params, **cfg)
    raise ValueError(f"Unsupported optimizer: {name}")

