from __future__ import annotations

from collections.abc import Callable, Mapping

BLOCK_REGISTRY: dict[str, Callable] = {}


def register_block(name: str):
    def decorator(obj):
        BLOCK_REGISTRY[str(name)] = obj
        return obj

    return decorator


def get_block(name: str):
    key = str(name)
    if key not in BLOCK_REGISTRY:
        raise KeyError(f"Unknown block: {key}. Available: {sorted(BLOCK_REGISTRY)}")
    return BLOCK_REGISTRY[key]


def build_block(config: Mapping | str, **kwargs):
    if isinstance(config, str):
        cls = get_block(config)
        return cls(**kwargs)
    if not isinstance(config, Mapping):
        raise TypeError("block config must be a mapping or string")
    cfg = dict(config)
    name = str(cfg.pop("name"))
    cls = get_block(name)
    cfg.update(kwargs)
    return cls(**cfg)
