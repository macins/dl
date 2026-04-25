from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("PyYAML is required to read YAML configs.")
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported config suffix: {suffix}")

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Top-level config must be a mapping, got {type(data)!r}.")
    return data


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_config_with_inheritance(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    cfg = load_config(path)
    inherits = cfg.pop("inherits", None)
    if inherits is None:
        return cfg

    parents = [inherits] if isinstance(inherits, (str, Path)) else list(inherits)
    merged: dict[str, Any] = {}
    for item in parents:
        parent = Path(item)
        if not parent.is_absolute():
            parent = (path.parent / parent).resolve()
        merged = _deep_merge(override=merged, base=load_config_with_inheritance(parent))
    return _deep_merge(merged, cfg)

