from __future__ import annotations

from typing import Mapping

from .base import BaseModel
from .registry import get_model


def build_model(config: Mapping | str | BaseModel) -> BaseModel:
    if isinstance(config, BaseModel):
        return config
    if isinstance(config, str):
        cls = get_model(config)
        return cls()
    if not isinstance(config, Mapping):
        raise TypeError("model config must be a mapping, string, or BaseModel")
    cfg = dict(config)
    name = str(cfg.pop("name"))
    cls = get_model(name)
    return cls(**cfg)

