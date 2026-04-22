import os

from datetime import time

import numpy as np
import polars as pl

import sys
sys.path.append("/root/auto-dl/jump")

from .utils import minus_time, get_data, add_se
from config import get_session_expr

ID_COLS = ["Time", "Symbol", "DATA_sector"]

def get_imb_expr(n1, n2):
    return (
        (pl.col(n1) - pl.col(n2)) / ((pl.col(n1) + pl.col(n2)).abs() + 1e-12)
    ).alias(f"feature_imb_{n1.lstrip("feature_")}_{n2.lstrip("feature_")}")

def neutralize(df, features=None, zscore=True):
    if features is None:
        features = [c for c in df.columns if "feature" in c and "neutralized" not in c]
    df = add_se(df)
    df = df.with_columns([
        (
            (pl.col(c) - pl.col(c).mean().over(["Time", "DATA_sector"])) / 
            (pl.col(c).std().over(["Time", "DATA_sector"]).fill_nan(None).fill_null(1) + 1e-12)
        ).alias(f"{c}_neutralized")
        for c in features
    ])
    if zscore:
        df = df.with_columns([
            (
                (pl.col(c) - pl.col(c).mean().over(["Time"])) / 
                (pl.col(c).std().fill_nan(None).fill_null(1).over(["Time"]) + 1e-12)
            ).alias(f"{c}_neutralized")
            for c in features
        ])
    return df

def get_zscore(df, features=None, windows=[2,3,5,10,20,30], verbose=False):
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    if features is None:
        features = [c for c in df.columns if "feature" in c]
    return df.with_columns([
        (
            (pl.col(c) - pl.col(c).rolling_mean(w, min_samples=2).over("Symbol")) / 
            (pl.col(c).rolling_std(w, min_samples=2).over("Symbol") + 1e-9)
        ).alias(f"{c}_zscore{w}")
        for c in features
        for w in windows
    ])

def gen_amt_mom(df, periods=[1,2,3,5,10,30,60]):
    """
    Args:
    df: DataFrame with columns ["Time", "Symbol", "Amount", "Close", "DATA_sector"]
    periods: list consists of periods

    Return: DataFrame of features
    """
    if isinstance(df, pl.LazyFrame):
        df = df.select(["Time", "Symbol", "Amount", "Close", "DATA_sector"]).collect()
    df = df.with_columns([
        (
            pl.col("Amount") * (pl.col("Close").log() - pl.col("Close").log().shift(w).over("Symbol"))
        ).fill_null(0).alias(f"feature_amt_mom_{w}")
        for w in periods
    ])
    long_periods = [w for w in periods if w >= 10]
    short_periods = [w for w in periods if w <= 10]
    df = df.with_columns([
        (
            (pl.col(f"feature_amt_mom_{lw}") - pl.col(f"feature_amt_mom_{sw}")) / 
            (pl.col(f"feature_amt_mom_{lw}").abs() + pl.col(f"feature_amt_mom_{sw}").abs() + 1e-12)
        ).alias(f"feature_amt_mom_{lw}-{sw}")
        for lw, sw in [(llw, ssw) for llw in long_periods for ssw in short_periods if llw > ssw]
    ])
    features = [c for c in df.columns if "feature" in c]
    return df.select(ID_COLS + features)

def gen_mom(df, periods=[1,2,3,5,10,30,60]):
    """
    Args:
    df: DataFrame with columns ["Time", "Symbol", "Close", "DATA_sector"]
    periods: list consists of periods

    Return: DataFrame of features
    """
    if isinstance(df, pl.LazyFrame):
        df = df.select(["Time", "Symbol", "Close", "DATA_sector"]).collect()
    df = df.with_columns([
        (
            pl.col("Close").log() - pl.col("Close").log().shift(w).over("Symbol")
        ).fill_null(0).alias(f"feature_mom_{w}")
        for w in periods
    ])
    long_periods = [w for w in periods if w >= 10]
    short_periods = [w for w in periods if w <= 10]
    df = df.with_columns([
        (
            (pl.col(f"feature_mom_{lw}") - pl.col(f"feature_mom_{sw}")) / 
            (pl.col(f"feature_mom_{lw}").abs() + pl.col(f"feature_mom_{sw}").abs() + 1e-12)
        ).alias(f"feature_mom_{lw}-{sw}")
        for lw, sw in [(llw, ssw) for llw in long_periods for ssw in short_periods if llw > ssw]
    ])
    features = [c for c in df.columns if "feature" in c]
    return df.select(ID_COLS + features)

def gen_pos(df, periods=[2, 3, 5, 10, 30, 60]):
    """
    Args:
    df: DataFrame with columns ["Time", "Symbol", "Close", "DATA_sector"]

    Return: DataFrame of features
    """
    if isinstance(df, pl.LazyFrame):
        df = df.select(["Time", "Symbol", "Open", "Close", "High", "Low", "DATA_sector"]).collect()
    df = df.with_columns([
        pl.when(pl.col("High") > pl.col("Low")).then(
            (pl.col("Close") - pl.col("Open")) / 
            (pl.col("High") - pl.col("Low"))
        ).otherwise(0).alias("feature_body_ratio"),
        pl.when(pl.col("High") > pl.col("Low")).then(
            (pl.col("Close") - pl.col("Low")) / 
            (pl.col("High") - pl.col("Low"))
        ).otherwise(0).alias("feature_pos")
    ])
    df = df.with_columns([
        (
            (pl.col("Close") - pl.col("Low").rolling_min(w, min_samples=2).over("Symbol")) / 
            (pl.col("High").rolling_max(w, min_samples=2).over("Symbol") - 
             pl.col("Low").rolling_min(w, min_samples=2).over("Symbol") + 1e-9)
        ).fill_null(strategy="forward").alias(f"feature_pos_{w}")
        for w in periods
    ])
    features = [c for c in df.columns if "feature" in c]
    return df.select(ID_COLS + features)

def gen_min_path(df, periods=[2,3,5,10,20,30,45,60,90]): # need zscore; zscore window = 2, 3 is enough
    """
    Args:
    df: DataFrame with columns ["Time", "Symbol", "Close", "DATA_sector"]
    periods: list consists of periods

    Return: DataFrame of features
    """
    if isinstance(df, pl.LazyFrame):
        df = df.select(["Time", "Symbol", "Close", "DATA_sector"]).collect()
    df = df.with_columns([
        (pl.col("Close") - pl.col("Close").shift(1).over("Symbol")).fill_null(0).alias("DeltaClose")
    ]).with_columns([
        (
            (pl.col("Close").log() - pl.col("Close").log().shift(w).over("Symbol")).abs().fill_null(0) / 
            (pl.col("DeltaClose").abs().rolling_sum(w, min_samples=2) + 1e-9)
        ).alias(f"feature_min_path_{w}")
        for w in periods
    ])
    features = [c for c in df.columns if "feature" in c]
    return df.select(ID_COLS + features)

def gen_vol_imb(df, periods = [2,3,5,10,30,45,60]):
    """
    Args:
    df: DataFrame with columns ["Time", "Symbol", "Close", "DATA_sector"]
    periods: list consists of periods

    Return: DataFrame of features
    """
    if isinstance(df, pl.LazyFrame):
        df = df.select(["Time", "Symbol", "Open", "Close", "Volume", "DATA_sector"]).collect()
    df = df.with_columns([
        (pl.col("Close") - pl.col("Open")).sign().alias("direction")
    ]).with_columns([
        (
            (pl.col("Volume") * pl.col("direction")).over("Symbol").rolling_sum(w, min_samples=2) / 
            (pl.col("Volume").over("Symbol").rolling_sum(w, min_samples=2) + 1e-9)
        ).alias(f"feature_imb_vol_{w}min")
        for w in periods
    ])
    long_periods = [w for w in periods if w >= 10]
    short_periods = [w for w in periods if w <= 10]
    df = df.with_columns([
        (
            (pl.col(f"feature_imb_vol_{lw}min") - pl.col(f"feature_imb_vol_{sw}min")) / 
            (pl.col(f"feature_imb_vol_{lw}min").abs() + pl.col(f"feature_imb_vol_{sw}min").abs() + 1e-9)       
        ).clip(-1,1).alias(f"feature_imb_vol_{lw}-{sw}min")
        for lw, sw in [(llw, ssw) for llw in long_periods for ssw in short_periods if llw > ssw]
    ])
    features = [c for c in df.columns if "feature" in c]
    return df.select(ID_COLS + features)

def gen_corr_CS(df, periods=[3,5,10,30,60]):
    """
    Args:
    df: DataFrame with columns ["Time", "Symbol", "Close", "DATA_sector"]
    periods: list consists of periods

    Return: DataFrame of features
    """
    if isinstance(df, pl.LazyFrame):
        df = df.select(["Time", "Symbol", "Close", "DATA_sector"]).collect()
    df = df.with_columns([
        pl.col("Close").rolling_std(w, min_samples=2).fill_null(0).alias(f"Close_std{w}")
        for w in [2, 3, 5]
    ])
    df = df.with_columns([
        pl.rolling_corr(
            "Close", f"Close_std{T}", window_size=w, min_samples=3
        ).over("Symbol")
            .fill_nan(None)
            .fill_null(strategy="forward")
            .replace(float("inf"), 1)
            .replace(-float("inf"), -1)
            .clip(-1, 1)
            .alias(f"feature_corr_CS{T}_{w}")
        for w in periods
        for T in [2, 3, 5]
    ])
    features = [c for c in df.columns if "feature" in c]
    return df.select(ID_COLS + features)

__all__ = ["neutralize", "get_zscore", "gen_amt_mom", "gen_mom", "gen_pos", "gen_min_path", "gen_vol_imb", "gen_corr_CS"]