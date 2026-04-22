import numpy as np
import polars as pl

from config import DataConfig, get_session_expr
data_cfg = DataConfig()
SYMBOLS = data_cfg.SYMBOLS
EXPR_DAY_SESSIONS = data_cfg.EXPR_DAY_SESSIONS
EXPR_NIGHT_SESSIONS = data_cfg.EXPR_NIGHT_SESSIONS
in_trading_session = EXPR_DAY_SESSIONS | EXPR_NIGHT_SESSIONS

IS_LAST_30MIN = (
    (pl.col("DATA_is_day_session") != pl.col("DATA_is_day_session").shift(-30)) |
    (
        (pl.col("DATA_is_day_session")) & (pl.col("Time").dt.date() != pl.col("Time").dt.date().shift(-30))
    )
)

def analyze_signal(
    df: pl.DataFrame,
    signal_col: str = "y_pred",
    return_col: str = "ret_30min",
) -> pl.DataFrame:
    df = df.with_columns([
        pl.when(EXPR_DAY_SESSIONS).then(True)
            .when(EXPR_NIGHT_SESSIONS).then(False)
            .otherwise(None).alias("DATA_is_day_session"),
    ])
    df = df.filter(in_trading_session)
    df = df.filter(~IS_LAST_30MIN)

    df = df.with_columns([
        pl.when(pl.col("Time").dt.hour() < 20).then(pl.col("Time").dt.date())
            .when(pl.col("Time").dt.weekday() < 5).then(pl.col("Time").dt.date() + pl.duration(days=1))
            .otherwise(pl.col("Time").dt.date() + pl.duration(days=3)).alias("TradingDay")
    ])
    df = df.with_columns([
        pl.col("Time").dt.month().alias("month"),
        pl.col("Time").dt.year().alias("year"),
    ])
    ic_col = f"ic_{return_col}"
    df_corr = df.with_columns([
        pl.col("Time").dt.year().alias("period")
    ]).group_by("period").agg([
        pl.corr(return_col, signal_col).alias("yearly_ic")
    ]).with_columns([
        ("(" + pl.col("period").cast(pl.String) + ",)").alias("period")
    ])
    df = df.group_by(["year", "month"]).agg([
        pl.corr(signal_col, return_col).alias(f"ic_{return_col}")
    ])
    yearly_results = []
    for year, df_group in df.group_by("year"):
        ic_summary = df_group.select([
            (pl.col(ic_col).mean() / (pl.col(ic_col).std() + 1e-12)).alias("yearly_ir"),
            pl.col(ic_col).mean().alias("ic_mean")
        ])
        yearly_results.append(ic_summary.with_columns(pl.lit(str(year)).alias("period")))
        
    final_report = pl.concat(yearly_results, how="diagonal").join(df_corr, on="period", how="left")
    return final_report.sort("period")