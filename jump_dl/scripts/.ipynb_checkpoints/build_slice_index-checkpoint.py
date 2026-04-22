from __future__ import annotations

import argparse

from jump_dl.src.dataio import build_symbol_day_index_cache


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
    out = build_symbol_day_index_cache(
        args.data_path,
        sorted_data_path=args.sorted_data_path,
        index_cache_path=args.index_cache_path,
        time_col=args.time_col,
        symbol_col=args.symbol_col,
        overwrite=args.overwrite,
    )
    print(out, flush=True)


if __name__ == "__main__":
    main()

