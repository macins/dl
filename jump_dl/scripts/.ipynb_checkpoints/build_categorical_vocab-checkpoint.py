from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build categorical vocab artifact.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--categorical-cols", required=True, help="Comma-separated categorical columns.")
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    categorical_cols = [c.strip() for c in args.categorical_cols.split(",") if c.strip()]
    lf = pl.scan_parquet(args.data_path)

    payload = {}
    for col in categorical_cols:
        counts = (
            lf.group_by(col)
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .collect()
        )
        values = counts[col].to_list()
        freq = counts["count"].to_list()
        token_to_id = {"__null__": 0}
        next_id = 2
        for value in values:
            if value is None:
                continue
            token_to_id[str(value)] = next_id
            next_id += 1
        payload[col] = {
            "padding_token_id": 0,
            "unknown_token_id": 1,
            "values": [None if v is None else str(v) for v in values],
            "counts": [int(v) for v in freq],
            "token_to_id": token_to_id,
            "vocab_size": next_id,
        }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()

