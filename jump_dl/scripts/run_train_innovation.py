from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import json

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jump_dl.src.config import load_config_with_inheritance


def _deep_merge_dict(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], Mapping) and isinstance(value, Mapping):
            out[key] = _deep_merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run jump_dl training with innovation adapter enabled.")
    p.add_argument("--config", required=True, help="Base config path (can use inheritance).")
    p.add_argument("--run-name", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--no-timestamp", action="store_true")

    p.add_argument("--innovation-aux-weight", type=float, default=0.01)
    p.add_argument("--innovation-prior-type", type=str, default="gru")
    p.add_argument("--innovation-fusion-type", type=str, default="mlp")
    p.add_argument("--innovation-min-log-s", type=float, default=-6.0)
    p.add_argument("--innovation-max-log-s", type=float, default=4.0)
    p.add_argument("--innovation-detach-aux-target", action="store_true", default=True)
    p.add_argument("--no-innovation-detach-aux-target", action="store_false", dest="innovation_detach_aux_target")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config_with_inheritance(args.config)

    override = {
        "model": {
            "innovation": {
                "enabled": True,
                "prior_type": args.innovation_prior_type,
                "fusion_type": args.innovation_fusion_type,
                "aux_loss_weight": args.innovation_aux_weight,
                "min_log_s": args.innovation_min_log_s,
                "max_log_s": args.innovation_max_log_s,
                "detach_aux_target": bool(args.innovation_detach_aux_target),
            }
        }
    }
    merged = _deep_merge_dict(cfg, override)

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        json.dump(merged, f, indent=2)
        temp_cfg = Path(f.name)

    cmd = [sys.executable, "jump_dl/scripts/run_train.py", "--config", str(temp_cfg)]
    if args.run_name is not None:
        cmd += ["--run-name", args.run_name]
    if args.output_dir is not None:
        cmd += ["--output-dir", args.output_dir]
    if args.no_timestamp:
        cmd += ["--no-timestamp"]

    try:
        subprocess.run(cmd, check=True)
    finally:
        temp_cfg.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
