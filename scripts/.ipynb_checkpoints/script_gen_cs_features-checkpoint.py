import os
import argparse
import multiprocessing
from src.features import neutralize
from src.feature_gen import ta_map
from src.feature_selection import feature_test, feature_test_symbol_wise, feature_selection
from src.preprocessor import CustomScaler
from config import DataConfig
from tqdm.auto import tqdm
import gc
import queue
import traceback
data_cfg = DataConfig()
SYMBOLS = data_cfg.SYMBOLS

import polars as pl
PERIOD = (pl.col("Time").dt.year() <= 2017) & (pl.col("Time").dt.year() >= 2015)

def save_memory_scale_task(args):
    s, df_sym, cols = args
    if len(df_sym.filter(PERIOD)) < 2:
        return s, df_sym
    scaler = CustomScaler(features=cols, fillna=True, clip=True, demean=True, scale=False, verbose=False)
    _ = scaler.fit_transform(df_sym.filter(PERIOD))
    return s, scaler.transform(df_sym)

def gen_one_feature(name, N=4):
    if os.path.exists(f"data/features/agg/{name}.parquet"):
        return
    if name in ["expma", "rsi"]:
        return
    y = pl.scan_parquet(f"data/labels/*.parquet").select(["Time", "Symbol", "ret_30min"]).collect().fill_nan(None).fill_null(0)
    df = pl.scan_parquet(f"data/features/{name}/*.parquet").fill_nan(None).fill_null(strategy="forward").fill_null(0)
    feature_cols = [c for c in df.collect_schema().names() if "feature" in c]
    T = len(feature_cols) // N + 1 
    if (len(feature_cols) % N) == 0:
        T -= 1
    ir_dict = {}
    final_features = []
    print(f"There are total {len(feature_cols)} {name} features and total {T} rounds...")
    for i in tqdm(range(T), position=0, leave=False):
        cur_dfs = {}
        cur_cols = feature_cols[N * i: N * (i + 1)]
        # print(cur_cols)
        cur_df = df.select(["Time", "Symbol", "DATA_sector"] + cur_cols).collect()
        print("Neutralizing features...")
        cur_df = neutralize(cur_df)
        cur_cols_n = [c for c in cur_df.columns if "feature" in c]

        ctx = multiprocessing.get_context("spawn")
        tasks = []
        for s in SYMBOLS:
            df_s = cur_df.filter(pl.col("Symbol") == s)
            tasks.append((s, df_s, cur_cols_n))
        
        with ctx.Pool(processes=10) as pool:
            for s, df_scaled in tqdm(
                pool.imap_unordered(save_memory_scale_task, tasks),
                total=len(tasks),
                position=1,
                leave=False,
                desc="Handling outliers..."
            ):
                cur_dfs[s] = df_scaled
        
        cur_df = pl.concat(cur_dfs.values())
        
        df_y = y.join(cur_df, on=["Time", "Symbol"], how="left")
        _, cur_ir_dict, selected = feature_test_symbol_wise(
            df=df_y,
            features=cur_cols_n,
            target="ret_30min",
            t_th=3,
            need_figure=False
        )
        _, _, _, selected = feature_test(
            df=df_y,
            features=selected,
            need_figure=False,
            t_th=3,
        )
        ir_dict.update(cur_ir_dict)
        if len(selected) >= 2:
            cur_final_features = feature_selection(
                df=cur_df.select(["Time", "Symbol"] + selected),
                features=selected,
                ir_dict=ir_dict,
                corr_th=0.7,
                need_figure=False
            )
        else:
            cur_final_features = selected
        final_features += cur_final_features
        
        cur_df = cur_df.select(["Time", "Symbol"] + cur_final_features)
        if i == 0:
            res_df = cur_df
        else:
            res_df = res_df.join(cur_df, on=["Time", "Symbol"], how="left")
    final_features = feature_selection(df=res_df, features=final_features, ir_dict=ir_dict, need_figure=False)
    os.makedirs(f"data/features/agg", exist_ok=True)
    res_df.select(["Time", "Symbol"] + final_features).write_parquet(f"data/features/agg/{name}.parquet")
    print(f"Final features are: {final_features}.\n")

def run_feature_gen(names, max_processes=1):
    processes = []
    for n in names:
        if len(processes) >= max_processes:
            for p in processes:
                p.join()
            processes = [p for p in processes if p.is_alive()]
            
        process = multiprocessing.Process(target=gen_one_feature, args=(n,))
        processes.append(process)
        process.start()

        gc.collect()
    for p in processes:
        p.join()
    return

def parse_args():
    parser = argparse.ArgumentParser(description="Generate cross-sectional features.")
    parser.add_argument(
        "--max_processes", type=int, default=1, help="Maximum number of concurrent processes."
    )
    return parser.parse_args()
    
if __name__ == "__main__":
    # names = ["amt_mom", "mom", "pos", "vol_imb", "corr_CS"]
    names = list(ta_map.keys())[4:]
    args = parse_args()
    run_feature_gen(names, args.max_processes)