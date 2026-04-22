import polars as pl
import os
from .features import *
from .talib import *
from functools import partial

group_col="Symbol"
ta_map = {
    "macd": partial(add_macd, fast=12, slow=26, signal=9, group_col=group_col),
    "dmi": partial(add_dmi, windows=[3, 5, 10, 20, 60], group_col=group_col),
    "dma": partial(add_dma, close="Close", short=10, long=50, ama_window=10, group_col=group_col),
    "expma": partial(add_expma, windows=[3, 5, 10, 20, 60], close="Close", group_col=group_col),
    "trix": partial(add_trix, close="Close", n=12, signal=9, group_col=group_col),
    "brar": partial(add_brar, windows=[3, 5, 10, 20, 60], group_col=group_col),
    "cr": partial(add_cr, windows=[3, 5, 10, 20, 60], group_col=group_col),
    "vr": partial(add_vr, windows=[3, 5, 10, 20, 60], group_col=group_col),
    "obv": partial(add_obv, group_col=group_col),
    "rsi": partial(add_rsi, windows=[3, 5, 10, 20, 60], close="Close", group_col=group_col),
    "wpr": partial(add_wpr, windows=[3, 5, 10, 20, 60], group_col=group_col),
    "kdj": partial(add_kdj, n=9, m1=3, m2=3, group_col=group_col),
    "cci": partial(add_cci, windows=[3, 5, 10, 20, 60], group_col=group_col),
    "roc": partial(add_roc, windows=[3, 5, 10, 20, 60], close="Close", group_col=group_col),
    "boll": partial(add_bollinger, windows=[3, 5, 10, 20, 60], close="Close", k=2.0, group_col=group_col),
    "turn": partial(add_turn, windows=[3, 5, 10, 20, 60], group_col=group_col),
    "bias": partial(add_bias, windows=[3, 5, 10, 20, 60], close="Close", group_col=group_col),
    "mtm": partial(add_mtm, windows=[3, 5, 10, 20, 60], close="Close", group_col=group_col),
    "psy": partial(add_psy, windows=[3, 5, 10, 20, 60], close="Close", group_col=group_col),
    "osc": partial(add_osc, windows=[3, 5, 10, 20, 60], close="Close", group_col=group_col),
    "b3612": partial(add_b3612, close="Close", n1=3, n2=6, n3=12, group_col=group_col),
}

def get_data(s: str, lazy: bool=False):
    if lazy:
        return pl.scan_parquet(f"data/cleaned_data/{s}.parquet")
    else:
        return pl.read_parquet(f"data/cleaned_data/{s}.parquet")

def get_feature(n: str, lazy: bool=True):
    if lazy:
        return pl.scan_parquet(f"data/features/{n}/*.parquet")
    else:
        return pl.read_parquet(f"data/features/{n}/*.parquet")

class TSFeatureGen:
    def __init__(self, symbol, need_save=True, use_f32=True):
        self.symbol = symbol
        self.data = get_data(self.symbol, lazy=True)
        self.use_f32 = use_f32
        self.need_save = need_save
        self.features = {}
        self.f = {}
        self.f["amt_mom"] = gen_amt_mom
        self.f["mom"] = gen_mom
        self.f["pos"] = gen_pos
        self.f["min_path"] = gen_min_path
        self.f["vol_imb"] = gen_vol_imb
        self.f["corr_CS"] = gen_corr_CS
        self.f.update(ta_map)
    
    def gen(self, name):
        if name not in self.f.keys():
            print(f"Congrats! You've invented the new feature {name.upper()}! Unfortunately, I don't know how to calculate it.")
            return
        self.features[name] = self.f[name](self.data)
        self.features[name] = get_zscore(self.features[name])
        if self.need_save:
            self.save(name)
        return self.features[name]

    def save(self, name):
        if name not in self.features.keys():
            print(f"Congrats! You've invented the new feature {name.upper()}! Unfortunately, I don't know how to calculate it.")
            return
        os.makedirs(f"data/features/{name}", exist_ok=True)
        feature_cols = [c for c in self.features[name].columns if "feature" in c]
        idx_cols = ["Time", "Symbol", "DATA_sector"]
        self.features[name] = self.features[name].select(idx_cols + feature_cols)
        if self.use_f32:
            self.features[name].with_columns([
                pl.selectors.numeric().cast(pl.Float32)
            ]).write_parquet(f"data/features/{name}/{self.symbol}.parquet", compression="zstd")
        else:
            self.features[name].write_parquet(f"data/features/{name}/{self.symbol}.parquet", compression="zstd")
        return