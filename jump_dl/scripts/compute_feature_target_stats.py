from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def _is_numeric_dtype(dtype: pl.DataType) -> bool:
    return dtype.is_numeric()


def _infer_columns(schema: pl.Schema, *, time_col: str, symbol_col: str) -> list[str]:
    reserved = {time_col, symbol_col}
    out = []
    for col in schema.names():
        if col in reserved:
            continue
        if _is_numeric_dtype(schema[col]):
            out.append(col)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute yearly feature/target stats for later merging.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--time-col", default="Time")
    parser.add_argument("--symbol-col", default="Symbol")
    parser.add_argument("--target-col", default="ret_30min")
    parser.add_argument("--columns", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schema = pl.scan_parquet(args.data_path).collect_schema()
    columns = [c.strip() for c in args.columns.split(",") if c.strip()]
    if not columns:
        columns = _infer_columns(schema, time_col=args.time_col, symbol_col=args.symbol_col)

    lf = pl.scan_parquet(args.data_path).with_columns(pl.col(args.time_col).dt.year().alias("year"))
    rows: list[pl.DataFrame] = []

    for col in columns:
        yearly = (
            lf.group_by("year")
            .agg([
                pl.col(col).is_not_null().sum().alias("count"),
                pl.col(col).fill_nan(None).drop_nulls().sum().alias("sum"),
                (pl.col(col).fill_nan(None).drop_nulls() ** 2).sum().alias("sum_sq"),
                pl.col(col).min().alias("min"),
                pl.col(col).max().alias("max"),
                pl.col(col).is_null().sum().alias("null_count"),
            ])
            .collect()
            .with_columns([
                pl.lit(col).alias("column"),
                pl.lit("target" if col == args.target_col else "feature").alias("kind"),
                pl.col("count").cast(pl.Int64),
                pl.col("sum").cast(pl.Float64),
                pl.col("sum_sq").cast(pl.Float64),
                pl.col("min").cast(pl.Float64),
                pl.col("max").cast(pl.Float64),
                pl.col("null_count").cast(pl.Int64),
            ])
        )
        yearly = yearly.with_columns([
            (pl.col("sum") / pl.col("count")).alias("mean"),
            (
                (pl.col("sum_sq") / pl.col("count")) - (pl.col("sum") / pl.col("count")) ** 2
            ).clip(lower_bound=0.0).sqrt().alias("std"),
        ])
        rows.append(yearly)

    out = pl.concat(rows, how="vertical").select([
        "year",
        "column",
        "kind",
        "count",
        "sum",
        "sum_sq",
        "min",
        "max",
        "null_count",
        "mean",
        "std",
    ]).sort(["year", "kind", "column"])

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(output_path)
    print(f"[done] wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
