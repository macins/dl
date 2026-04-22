from __future__ import annotations

from typing import Mapping

from .utils.externals import ensure_torch

torch = ensure_torch()


def build_scheduler(config: Mapping | None, optimizer):
    if config is None:
        return None
    cfg = dict(config)
    name = str(cfg.pop("name", "cosine")).lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **cfg)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, **cfg)
    raise ValueError(f"Unsupported scheduler: {name}")

