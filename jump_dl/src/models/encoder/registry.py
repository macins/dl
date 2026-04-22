from __future__ import annotations

from collections.abc import Callable, Mapping

ENCODER_REGISTRY: dict[str, Callable] = {}


def register_encoder(name: str):
    def decorator(obj):
        ENCODER_REGISTRY[str(name)] = obj
        return obj

    return decorator


def get_encoder(name: str):
    key = str(name)
    if key not in ENCODER_REGISTRY:
        raise KeyError(f"Unknown encoder: {key}. Available: {sorted(ENCODER_REGISTRY)}")
    return ENCODER_REGISTRY[key]


def build_encoder(config: Mapping | str, **kwargs):
    if isinstance(config, str):
        cls = get_encoder(config)
        return cls(**kwargs)
    if not isinstance(config, Mapping):
        raise TypeError("encoder config must be a mapping or string")
    cfg = dict(config)
    name = str(cfg.pop("name"))
    cls = get_encoder(name)
    cfg.update(kwargs)
    return cls(**cfg)
