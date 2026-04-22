from datetime import datetime, timedelta
import numpy as np
import polars as pl

def minus_time(t, d):
    return (datetime.combine(datetime.today().date(), t) - timedelta(minutes=d)).time()

import os, sys
PROJ_ROOT = "/root/autodl-tmp/jump"
sys.path.append(PROJ_ROOT)
from config import DataConfig
data_cfg = DataConfig()
SYMBOLS = data_cfg.SYMBOLS

sector = {}
exchange = {}
for s in SYMBOLS:
    if s in ["RB", "HC", "SS", "I", "J", "JM", "SF", "SM"]:
        sector[s] = "heise"    # 黑色系
    elif s in ["AU", "AG"]:
        sector[s] = "guijinshu"    # 贵金属
    elif s in ["CU", "AL", "ZN", "PB", "NI", "SN"]:
        sector[s] = "yousejinshu"    # 有色金属
    elif s in ["SC", "FU", "BU", "PG", "EG", "EB", "L", "V", "PP", "MA", "TA", "SA", "UR", "RU", "NR", "SP", "ZC"]:
        sector[s] = "nengyuanhuagong"    # 能源化工
    elif s in ["A", "B", "M", "Y", "P", "OI", "RM", "C", "CS", "RR", "CF", "CY", "AP", "CJ", "JD", "SR"]:
        sector[s] = "nongfuchanpin"    # 农副产品
    elif s in ["FB", "FG", "SP", "BP", "LW", "GP"]:
        sector[s] = "qinggongjiancai"    # 轻工建材
    else:
        print(s, " not included in sector classification!")

    if s in ["SC", "NR"]:
        exchange[s] = "INE"    # 上期能源
    elif s in ["CU", "AL", "ZN", "PB", "NI", "SN", "AU", "AG", "RB", "HC", "SS", "FU", "BU", "RU", "SP"]:
        exchange[s] = "SHFE"    # 上期所
    elif s in ["I", "J", "JM", "L", "V", "PP", "EG", "EB", "PG", "C", "CS", "A", "B", "M", "Y", "P", "RR", "FB", "JD"]:
        exchange[s] = "DCE"    # 大商所
    elif s in ["CF", "CY", "SR", "AP", "CJ", "OI", "RM", "TA", "MA", "SA", "FG", "ZC", "UR", "SF", "SM"]:
        exchange[s] = "CZCE"    # 郑商所
    else:
        print(s, " not included in exchange classification!")

def add_se(df):
    if "DATA_sector" not in df.columns:
        df = df.with_columns([
            pl.col("Symbol").replace_strict(sector).alias("DATA_sector")
        ])
    if "DATA_exchange" not in df.columns:
        df = df.with_columns([
            pl.col("Symbol").replace_strict(exchange).alias("DATA_exchange")
        ])
    return df

def get_raw_data(s: str, lazy: bool=False):
    if lazy:
        return pl.scan_csv(f"{PROJ_ROOT}/data/raw_data/{s}.csv")
    else:
        return pl.read_csv(f"{PROJ_ROOT}/data/raw_data/{s}.csv")

def get_corr(x, y):
    return np.corrcoef(x,y)[0,1]

def get_data(s: str, lazy: bool=False):
    if lazy:
        return pl.scan_parquet(f"{PROJ_ROOT}/data/cleaned_data/{s}.parquet")
    else:
        return pl.read_parquet(f"{PROJ_ROOT}/data/cleaned_data/{s}.parquet")

def get_ret(s: str, lazy: bool=False, rets=[30]):
    if lazy:
        return pl.scan_parquet(f"{PROJ_ROOT}/data/labels/{s}.parquet").select(["Time", "Symbol"] + [f"ret_{w}min" for w in rets])
    else:
        return pl.scan_parquet(f"{PROJ_ROOT}/data/labels/{s}.parquet").select(["Time", "Symbol"] + [f"ret_{w}min" for w in rets]).collect()

def get_feature(s: str, n: str, lazy: bool=False):
    if lazy:
        return pl.scan_parquet(f"{PROJ_ROOT}/data/features/{n}/{s}.parquet")
    else:
        return pl.read_parquet(f"{PROJ_ROOT}/data/features/{n}/{s}.parquet")

def get_training_data(features=[], symbol=None):
    y = pl.scan_parquet(f"../results/LinearRegression/1120_2050.parquet")
    df = pl.scan_parquet(f"../data/cleaned_data/*.parquet").select(["Time", "Symbol", "DATA_sector", "DATA_is_day_session", "DATA_is_trading_hour", "DATA_is_last_30min"])
    for f in features:
        df = df.join(pl.scan_parquet(f"../data/features/agg/{f}.parquet"), on=["Time", "Symbol"], how="left")
        df = df.drop([c for c in df.collect_schema().names() if "_right" in c])
    df = df.fill_nan(None).fill_null(0)
    df = df.join(y, on=["Time", "Symbol"], how="left")
    if symbol is None:
        return df
    else:
        return df.filter(pl.col("Symbol") == symbol)