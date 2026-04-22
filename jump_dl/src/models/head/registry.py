from __future__ import annotations

from collections.abc import Callable, Mapping

HEAD_REGISTRY: dict[str, Callable] = {}


def register_head(name: str):
    def decorator(obj):
        HEAD_REGISTRY[str(name)] = obj
        return obj

    return decorator


def get_head(name: str):
    key = str(name)
    if key not in HEAD_REGISTRY:
        raise KeyError(f"Unknown head: {key}. Available: {sorted(HEAD_REGISTRY)}")
    return HEAD_REGISTRY[key]


def build_head(config: Mapping | str, **kwargs):
    if isinstance(config, str):
        cls = get_head(config)
        return cls(**kwargs)
    if not isinstance(config, Mapping):
        raise TypeError("head config must be a mapping or string")
    cfg = dict(config)
    name = str(cfg.pop("name"))
    cls = get_head(name)
    cfg.update(kwargs)
    return cls(**cfg)
