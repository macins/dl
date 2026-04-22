# scripts/build_final_features.py
import os
import re
import polars as pl

from src.features import (
    neutralize,
    gen_amt_mom,
    gen_mom,
    gen_pos,
    gen_min_path,
    gen_vol_imb,
    gen_corr_CS,
)
from src.talib import (
    add_mtm,
    add_roc,
    add_psy,
    add_rsi,
    add_cr,
    add_bollinger,
    add_dmi,
    add_b3612,
    add_vr,
    add_wpr,
    add_obv,
    add_brar,
)

ID_COLS = ["Time", "Symbol", "DATA_sector"]

CAT_FEATURES = [
    "Symbol",
    "DATA_sector",
    "DATA_month",
    "DATA_is_day_session"
]

FINAL_FEATURES = [
    "feature_mtm_5_zscore10_neutralized",
    "feature_roc_20",
    "feature_psy_5_zscore2_neutralized",
    "feature_rsi_10_zscore10",
    "feature_imb_vol_2min_zscore3_neutralized",
    "feature_mom_10",
    "feature_mom_30_zscore2_neutralized",
    "feature_roc_3_neutralized",
    "feature_cr_3_neutralized",
    "feature_mtm_3_zscore5_neutralized",
    "feature_bollinger_low_3_zscore5",
    "feature_imb_vol_10min",
    "feature_dmi_minus_di_60",
    "feature_imb_vol_2min",
    "feature_roc_20_zscore10_neutralized",
    "feature_b612_6_12_zscore2",
    "feature_corr_CS5_30",
    "feature_vr_10_neutralized",
    "feature_bollinger_up_5_zscore2",
    "feature_rsi_3_neutralized",
    "feature_corr_CS2_3",
    "feature_wpr_3_zscore3",
    "feature_mom_1_neutralized",
    "feature_bollinger_low_5_zscore10",
    "feature_mom_2_zscore2",
    "feature_corr_CS5_3",
    "feature_amt_mom_2_zscore3",
    "feature_obv_zscore10",
    "feature_amt_mom_1_zscore10",
    "feature_mtm_5_zscore3_neutralized",
    "feature_dmi_minus_di_3_zscore10_neutralized",
    "feature_mtm_10_zscore10_neutralized",
    "feature_b36_3_6_zscore5_neutralized",
    "feature_vr_5_neutralized",
    "feature_dmi_plus_di_20_zscore2_neutralized",
    "feature_bollinger_up_3_zscore10",
    "feature_mtm_60_zscore2_neutralized",
    "feature_bollinger_low_10_zscore2",
    "feature_bollinger_up_10_zscore2",
    "feature_mtm_20_zscore2_neutralized",
    "feature_psy_3_zscore2_neutralized",
    "feature_cr_5_neutralized",
    "feature_psy_5_zscore10",
    "feature_rsi_60",
    "feature_amt_mom_1",
    "feature_amt_mom_1_zscore3",
    "feature_bollinger_mid_5_zscore2",
    "feature_roc_3_zscore2",
    "feature_mtm_10_zscore2",
    "feature_corr_CS5_60",
    "feature_pos_2_zscore5_neutralized",
    "feature_amt_mom_1_neutralized",
    "feature_rsi_60_zscore30",
    "feature_mom_60",
    "feature_psy_3",
    "feature_bollinger_low_3_zscore2",
    "feature_corr_CS2_5",
    "feature_mom_5_neutralized",
    "feature_corr_CS5_10",
    "feature_psy_10",
    "feature_imb_vol_3min_zscore3_neutralized",
    "feature_corr_CS5_5",
    "feature_bollinger_up_3_zscore2",
    "feature_bollinger_low_5_zscore2",
    "feature_corr_CS3_5",
    "feature_wpr_5_zscore5",
    "feature_min_path_2_zscore2",
    "feature_corr_CS3_3",
    "feature_vr_3_neutralized",
    "feature_brar_br_10_zscore3",
]

NEUTRALIZE_BASE = sorted(
    {
        name.rsplit("_neutralized", 1)[0]
        for name in FINAL_FEATURES
        if name.endswith("_neutralized")
    }
)

_ZSCORE_RE = re.compile(r"^(.*)_zscore(\d+)$")


def _strip_neutralized(name: str) -> str:
    return name[:-12] if name.endswith("_neutralized") else name


def infer_required_raw_base_cols(final_features: list[str]) -> list[str]:
    """
    从 FINAL_FEATURES 反推出：
    1) 需要从 base.parquet 读取哪些原始 feature 列
    """
    raw_cols = set()

    for name in final_features:
        core = _strip_neutralized(name)
        m = _ZSCORE_RE.match(core)
        if m:
            raw_cols.add(m.group(1))   # e.g. feature_mtm_5 from feature_mtm_5_zscore10
        else:
            raw_cols.add(core)         # e.g. feature_roc_20 or feature_roc_3

    return sorted(raw_cols)


def infer_required_zscore_map(final_features: list[str]) -> dict[str, list[int]]:
    """
    从 FINAL_FEATURES 反推出：
    2) 只需要计算哪些精确的 (base_feature, z_window)
       返回形式:
       {
         "feature_mtm_5": [3, 10],
         "feature_rsi_10": [10],
         ...
       }
    """
    zmap: dict[str, set[int]] = {}

    for name in final_features:
        core = _strip_neutralized(name)
        m = _ZSCORE_RE.match(core)
        if not m:
            continue
        base_col = m.group(1)
        window = int(m.group(2))
        zmap.setdefault(base_col, set()).add(window)

    return {k: sorted(v) for k, v in sorted(zmap.items())}


def add_needed_zscores(
    df: pl.DataFrame,
    zscore_map: dict[str, list[int]],
    group_col: str = "Symbol",
) -> pl.DataFrame:
    """
    只对 zscore_map 指定的那些 (feature, window) 计算 rolling zscore。
    不做 feature x all_windows 的笛卡尔积展开。
    """
    if not zscore_map:
        return df

    missing = [c for c in zscore_map if c not in df.columns]
    if missing:
        raise ValueError(f"Missing base columns for zscore: {missing}")

    exprs = []
    for c, windows in zscore_map.items():
        for w in windows:
            exprs.append(
                (
                    (pl.col(c) - pl.col(c).rolling_mean(w, min_samples=2).over(group_col))
                    / (pl.col(c).rolling_std(w, min_samples=2).over(group_col) + 1e-9)
                ).alias(f"{c}_zscore{w}")
            )

    return df.with_columns(exprs)
    
# 需要做 neutralize 的底层列 = 去掉结尾的 `_neutralized`
NEUTRALIZE_BASE = sorted(
    {name.rsplit("_neutralized", 1)[0]
     for name in FINAL_FEATURES
     if name.endswith("_neutralized")}
)


# ==== 3. 读原始数据 ====
def load_raw_data(path_pattern: str = "data/cleaned_data/*.parquet") -> pl.DataFrame:
    """
    读入所有 symbol 的 cleaned_data，拼成一个 panel。
    这样 neutralize 可以在 (Time, DATA_sector) 截面上做。
    """
    print("Loading raw data...")
    return (
        pl.scan_parquet(path_pattern)
        .select([
            "Time",
            "Symbol",
            "DATA_sector",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Amount",
            "OpenInterest",
        ])
        .sort(["Symbol", "Time"])
        .collect()
    )


# ==== 4. 计算 TA 指标（只保留你会用到的那些 window） ====
def build_ta_features(df: pl.DataFrame) -> pl.DataFrame:
    lf = df.sort(["Symbol", "Time"]).lazy()

    print("Building ta features...")
    # 这些 window 都是从你最终特征里抽出来的
    lf = add_mtm(lf, windows=[3, 5, 10, 20, 60], close="Close", group_col="Symbol")
    lf = add_roc(lf, windows=[3, 20], close="Close", group_col="Symbol")
    lf = add_psy(lf, windows=[3, 5, 10], close="Close", group_col="Symbol")
    lf = add_rsi(lf, windows=[3, 10, 60], close="Close", group_col="Symbol")
    lf = add_cr(lf, windows=[3, 5], group_col="Symbol")
    lf = add_bollinger(lf, windows=[3, 5, 10], close="Close", k=2.0, group_col="Symbol")
    lf = add_dmi(lf, windows=[3, 20, 60], group_col="Symbol")
    lf = add_b3612(lf, close="Close", n1=3, n2=6, n3=12, group_col="Symbol")
    lf = add_vr(lf, windows=[3, 5, 10], group_col="Symbol")
    lf = add_wpr(lf, windows=[3, 5], group_col="Symbol")
    lf = add_obv(lf, group_col="Symbol")
    lf = add_brar(lf, windows=[10], group_col="Symbol")

    # 统一把 TA 里的 NaN/null 补成 0
    lf = lf.with_columns(
        pl.col("^feature_.*$").fill_nan(0.0).fill_null(0.0)
    )
    return lf.collect()


# ==== 5. 计算你自定义的 K 线特征 ====
def build_custom_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    用 features.py 里的函数算：
    amt_mom / mom / pos / min_path / vol_imb / corr_CS
    然后横向 concat。
    """
    print("Building custom features...")
    f_amt_mom   = gen_amt_mom(df)   # feature_amt_mom_*
    f_mom       = gen_mom(df)       # feature_mom_*
    f_pos       = gen_pos(df)       # feature_pos_*
    f_min_path  = gen_min_path(df)  # feature_min_path_*
    f_vol_imb   = gen_vol_imb(df)   # feature_imb_vol_*min
    f_corr_cs   = gen_corr_CS(df)   # feature_corr_CS*_*

    id_df = df.select(ID_COLS)

    others = [
        f_amt_mom.drop(ID_COLS),
        f_mom.drop(ID_COLS),
        f_pos.drop(ID_COLS),
        f_min_path.drop(ID_COLS),
        f_vol_imb.drop(ID_COLS),
        f_corr_cs.drop(ID_COLS),
    ]

    return pl.concat([id_df, *others], how="horizontal")


# ==== 6. 一次性拼出所有“基底特征” ====
def build_base_features(df: pl.DataFrame) -> pl.DataFrame:
    ta_df   = build_ta_features(df)
    cust_df = build_custom_features(df)

    # 两边都有 ID_COLS，而且行顺序一样（都从同一个 df 变换出来）
    id_df   = df.select(ID_COLS)
    ta_cols = ta_df.drop(ID_COLS)
    cs_cols = cust_df.drop(ID_COLS)

    base = pl.concat([id_df, ta_cols, cs_cols], how="horizontal")

    # 再兜底一次缺失值：所有 feature 列 NaN/null 都补成 0
    base = base.with_columns(
        pl.col("^feature_.*$").fill_nan(0.0).fill_null(0.0)
    )
    return base


def add_zscore_and_neutralize(df: pl.DataFrame) -> pl.DataFrame:
    print("Inferring required zscore specs...")
    zscore_map = infer_required_zscore_map(FINAL_FEATURES)
    n_zcols = sum(len(v) for v in zscore_map.values())
    print(f"Need only {n_zcols} zscore columns.")

    print("Adding only required zscores...")
    df = add_needed_zscores(df, zscore_map=zscore_map, group_col="Symbol")

    print("Neutralizing only required columns...")
    missing_for_neutralize = [c for c in NEUTRALIZE_BASE if c not in df.columns]
    if missing_for_neutralize:
        raise ValueError(
            f"Columns required by neutralize are missing: {missing_for_neutralize}"
        )

    if NEUTRALIZE_BASE:
        df = neutralize(df, features=NEUTRALIZE_BASE, zscore=True)

    df = df.with_columns(
        pl.col("^feature_.*$").fill_nan(0.0).fill_null(0.0)
    )
    return df


def main():
    os.makedirs("data/features", exist_ok=True)

    # 如果你还没生成 base.parquet，就打开下面这几行
    df_raw = load_raw_data("data/cleaned_data/*.parquet")
    df_base = build_base_features(df_raw)
    # df_base.write_parquet("data/features/base.parquet", compression="zstd")

    required_raw_cols = infer_required_raw_base_cols(FINAL_FEATURES)
    # print(f"Reading only required raw base columns: {len(required_raw_cols)}")

    # 关键优化：
    # 不要把整个 base.parquet 全读进来，只读 FINAL_FEATURES 真正依赖的列
    # df_base = (
    #     pl.scan_parquet("data/features/base.parquet")
    #     .select(ID_COLS + required_raw_cols)
    #     .collect()
    # )

    # 只计算需要的 zscore，再 neutralize
    df_full = add_zscore_and_neutralize(df_base)

    # 最终只保留提交所需列
    final_cols = ID_COLS + FINAL_FEATURES
    df_final = df_full.select(final_cols)

    labels = pl.read_parquet("data/labels/*.parquet")
    df_final = df_final.join(labels, on=["Time", "Symbol"], how="left")
    df_final = df_final.join(df_raw, on=["Time", "Symbol"], how="left")

    print("Computing done. Saving...")
    df_final.write_parquet(
        "data/features/data.parquet",
        compression="zstd",
    )
    print("Done. Saved to data/features/data.parquet")

if __name__ == "__main__":
    main()