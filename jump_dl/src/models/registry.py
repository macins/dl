from __future__ import annotations

from collections.abc import Callable

MODEL_REGISTRY: dict[str, Callable] = {}


def register_model(name: str):
    def decorator(obj):
        MODEL_REGISTRY[str(name)] = obj
        return obj

    return decorator


def get_model(name: str):
    key = str(name)
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {key}. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key]

