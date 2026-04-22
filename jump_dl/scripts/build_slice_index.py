from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sorted parquet and symbol-day slice index.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--sorted-data-path", required=True)
    parser.add_argument("--index-cache-path", required=True)
    parser.add_argument("--time-col", default="Time")
    parser.add_argument("--symbol-col", default="Symbol")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sorted_path = Path(args.sorted_data_path)
    index_path = Path(args.index_cache_path)
    if not args.overwrite and (sorted_path.exists() or index_path.exists()):
        raise FileExistsError("Output exists. Pass --overwrite to replace existing artifacts.")

    df = pl.read_parquet(args.data_path).with_columns(
        pl.col(args.time_col).dt.date().alias("__date__")
    ).sort(["__date__", args.symbol_col, args.time_col])
    df.write_parquet(sorted_path)

    indexed = df.with_row_index("__row_idx__")
    spans = (
        indexed.group_by(["__date__", args.symbol_col], maintain_order=True)
        .agg(
            pl.col("__row_idx__").min().alias("start"),
            (pl.col("__row_idx__").max() + 1).alias("end"),
            pl.len().alias("length"),
            pl.col(args.time_col).min().alias("time_min"),
            pl.col(args.time_col).max().alias("time_max"),
        )
        .rename({"__date__": "Date", args.symbol_col: "Symbol"})
    )
    spans.write_parquet(index_path)
    print(
        {
            "sorted_data_path": str(sorted_path),
            "index_cache_path": str(index_path),
            "num_rows": df.height,
            "num_slices": spans.height,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
