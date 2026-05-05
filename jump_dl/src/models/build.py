from __future__ import annotations

import inspect
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

    init_sig = inspect.signature(cls.__init__)
    params = init_sig.parameters
    accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    accepted_keys = {
        k
        for k, p in params.items()
        if k != "self" and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }

    if not accepts_var_kwargs:
        # Inherited config merges sometimes leave null markers for keys that are
        # specific to a different model family (e.g. `backbone: null`). Treat
        # those null unknowns as "unset", but keep strict errors for real values.
        for key in list(cfg.keys()):
            if key not in accepted_keys and cfg[key] is None:
                cfg.pop(key)

        unknown_keys = [key for key in cfg if key not in accepted_keys]
        if unknown_keys:
            raise TypeError(f"{cls.__name__} got unexpected config keys: {sorted(unknown_keys)}")

    return cls(**cfg)
