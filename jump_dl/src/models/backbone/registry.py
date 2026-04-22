from __future__ import annotations

from collections.abc import Callable, Mapping

BACKBONE_REGISTRY: dict[str, Callable] = {}


def register_backbone(name: str):
    def decorator(obj):
        BACKBONE_REGISTRY[str(name)] = obj
        return obj

    return decorator


def get_backbone(name: str):
    key = str(name)
    if key not in BACKBONE_REGISTRY:
        raise KeyError(f"Unknown backbone: {key}. Available: {sorted(BACKBONE_REGISTRY)}")
    return BACKBONE_REGISTRY[key]


def build_backbone(config: Mapping | str, **kwargs):
    if isinstance(config, str):
        cls = get_backbone(config)
        return cls(**kwargs)
    if not isinstance(config, Mapping):
        raise TypeError("backbone config must be a mapping or string")
    cfg = dict(config)
    name = str(cfg.pop("name"))
    cls = get_backbone(name)
    cfg.update(kwargs)
    return cls(**cfg)
