from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def serialize_vocab_key(value: Any) -> str:
    if value is None:
        return "__null__"
    return str(value)


def load_vocab(path: str | Path) -> dict[str, dict[str, int]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, dict[str, int]] = {}
    for column, info in payload.items():
        token_to_id = info.get("token_to_id", {})
        out[str(column)] = {str(k): int(v) for k, v in token_to_id.items()}
    return out

