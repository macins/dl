from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl


@dataclass
class ColumnStats:
    mean: np.ndarray
    std: np.ndarray
    columns: list[str]
    count: np.ndarray | None = None


def _load_stats_frame(stats_path: str | Path) -> pl.DataFrame:
    path = Path(stats_path)
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found under stats dir: {path}")
        return pl.concat([pl.read_parquet(p) for p in files], how="vertical")
    return pl.read_parquet(path)


def merge_yearly_stats(
    stats_path: str | Path,
    *,
    years: Sequence[int] | None = None,
    columns: Sequence[str] | None = None,
    kind: str = "feature",
    eps: float = 1e-6,
) -> ColumnStats:
    frame = _load_stats_frame(stats_path).filter(pl.col("kind") == kind)
    if years is not None:
        frame = frame.filter(pl.col("year").is_in([int(v) for v in years]))
    if columns is not None:
        frame = frame.filter(pl.col("column").is_in(list(columns)))

    grouped = (
        frame.group_by("column")
        .agg([
            pl.col("count").sum().alias("count"),
            pl.col("sum").sum().alias("sum"),
            pl.col("sum_sq").sum().alias("sum_sq"),
        ])
        .sort("column")
    )
    if grouped.height == 0:
        raise ValueError("No stats rows matched the requested filters.")

    cols = grouped["column"].to_list()
    count = grouped["count"].to_numpy().astype(np.float64)
    total_sum = grouped["sum"].to_numpy().astype(np.float64)
    total_sum_sq = grouped["sum_sq"].to_numpy().astype(np.float64)

    mean = total_sum / np.maximum(count, 1.0)
    var = np.maximum(total_sum_sq / np.maximum(count, 1.0) - mean**2, 0.0)
    std = np.sqrt(var)
    std = np.where(std < eps, 1.0, std)
    return ColumnStats(
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        columns=cols,
        count=count.astype(np.float64),
    )

