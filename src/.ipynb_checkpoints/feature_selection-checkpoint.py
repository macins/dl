import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

import fastcluster
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

import os
import sys
sys.path.append(os.getcwd().rsplit("/", 1)[0])

from config import DataConfig
data_cfg = DataConfig()
SYMBOLS = data_cfg.SYMBOLS

def NW_var(c: pl.Series):
    import numpy as np
    T = len(c)
    L = max(0, min(int(4 * (T / 100) ** (2 / 9)), T - 1))
    c = c.fill_nan(None).fill_null(strategy="forward").fill_null(0)
    mu = c.mean()
    cov = [( (c - mu) * (c - mu).shift(lag) ).fill_null(0).mean() * T / (T - lag) for lag in range(L+1)]
    cov_np = np.array(cov[1:])
    w = np.array([2 * (1 - l / (L+1)) for l in range(1, L+1)])
    return cov[0] + (w * cov_np).sum()

def feature_test(
    df: pl.LazyFrame | pl.DataFrame, 
    features=[], 
    target: str="ret_30min",
    ic_th: float=0.00,
    t_th: float=3,
    need_figure: bool=True,
    to_skip=[],
    raw_vs_neu: bool=True
):
    if isinstance(df, pl.LazyFrame):
        cur_cols = df.collect_schema().names()
    elif isinstance(df, pl.DataFrame):
        cur_cols = df.columns

    if len(features) == 0:
        features = [c for c in cur_cols if "feature" in c]

    df = df.filter((pl.col("Time").dt.year() < 2021) & (pl.col("Time").dt.year() > 2013))

    sign_df = df.select([
        pl.corr(target, c).sign().alias(c)
        for c in features
    ]).unpivot()
    signs = dict(zip(sign_df["variable"], sign_df["value"]))
    
    df = df.with_columns([
        (pl.col(c) * signs[c]).alias(c)
        for c in features
    ])
    
    df = df.with_columns([
        pl.col("Time").dt.year().alias("year"),
        pl.col("Time").dt.month().alias("month"),
        pl.col("Time").dt.day().alias("day")
    ]).fill_nan(None).fill_null(0)

    ic_df = df.group_by(["year", "month", "day"]).agg([
        pl.corr(
            pl.col(c),
            pl.col(target)
        ).alias(c)
        for c in features
    ]).fill_nan(None).fill_null(0)

    if isinstance(ic_df, pl.LazyFrame):
        ic_df = ic_df.collect()
        
    ic_df = ic_df.with_columns([
        (
            pl.col("year").cast(pl.String) + "-" + 
             pl.col("month").cast(pl.String).str.zfill(2) + "-" +
             pl.col("day").cast(pl.String).str.zfill(2)
        ).str.strptime(pl.Date, "%Y-%m-%d").alias("time")
    ])
    
    ic_df = ic_df.sort("time")
    long = ic_df.drop("time").unpivot(
        index = ["year", "month", "day"],
        variable_name = "var",
        value_name = "value",
    )

    y = long.group_by(["year", "var"]).agg([
        pl.len().alias("T"),
        pl.col("value").mean().alias("ic_mean"),
        (
            pl.col("value").mean() /
            (pl.col("value").std() + 1e-12)
        ).alias("ir"),
        (
            (pl.col("value").mean() - ic_th) /
            (pl.col("value").map_batches(
                lambda arr: NW_var(arr),
                return_dtype=pl.Float64,
                returns_scalar=True
            ) / (pl.len() + 1e-12)).sqrt()
        ).alias("t-stats"),
        (
            (pl.col("value").mean() - ic_th) /
            (pl.col("value").std() + 1e-12) *
            pl.len().sqrt()
        ).alias("naive-t-stats"),
    ])

    m = long.group_by(["year", "month", "var"]).agg([
        pl.len().alias("T"),
        pl.col("value").mean().alias("ic_mean"),
        (
            pl.col("value").mean() /
            (pl.col("value").std() + 1e-12)
        ).alias("ir"),
        (
            (pl.col("value").mean() - ic_th) /
            (pl.col("value").map_batches(
                lambda arr: NW_var(arr),
                return_dtype=pl.Float64,
                returns_scalar=True
            ) / (pl.len() + 1e-12)).sqrt()
        ).alias("t-stats"),
        (
            (pl.col("value").mean() - ic_th) /
            (pl.col("value").std() + 1e-12) *
            pl.len().sqrt()
        ).alias("naive-t-stats"),
    ])
    
    m = m.with_columns(
        (pl.col("year").cast(pl.String) + "-" + pl.col("month").cast(pl.String).str.zfill(2)).alias("ts")
    ).sort("ts")
    
    # print(f"Selecting features with t-stats > {t_th}...")

    selected = []
    for c in features:
        t_stats = y.filter(pl.col("var")==c)["t-stats"].to_numpy()
        pos = np.sum(t_stats > 0)
        neg = np.sum(t_stats < 0)
        if neg >= 1:
            continue
        t_cri = np.delete(t_stats, t_stats.argmax()).mean()
        if t_cri < t_th:
            continue
        if np.median(t_stats) < t_th:
            continue
        ic_list = ic_df[c]
        t_all = ic_list.mean() / np.sqrt(NW_var(ic_list)) * np.sqrt(len(ic_list))
        if t_all < t_th:
            continue
        selected.append(c)

    ir_df = ic_df.filter(
        pl.col("time").dt.year() < 2018
    ).select([
        (pl.col(c).mean() / (pl.col(c).std() + 1e-12)).alias(c)
        for c in features
    ]).unpivot().rename({"variable": "feature", "value": "ir"})

    ir_dict = dict(zip(ir_df["feature"], ir_df["ir"]))
    if need_figure and raw_vs_neu:
        if len(to_skip) == 0:
            to_skip = [c for c in features if "neutralized" in c]
        for c in features:
            if c in to_skip:
                continue
            fig = plt.figure(figsize=(24,24))
            gs = fig.add_gridspec(5,2)
            ax = [[None, None] for _ in range(5)]
            for i in range(5):
                if i == 4:
                    continue
                for j in range(2):
                    ax[i][j] = fig.add_subplot(gs[i,j])
            ax[4] = fig.add_subplot(gs[4,:])
            tmp_m = m.filter(pl.col("var") == c).sort("ts")
            
            ax[0][0].bar(
                x = tmp_m["ts"],
                height = tmp_m["t-stats"]
            )
            ax[0][0].set_xticks(range(0, len(tmp_m["ts"]), 12))
            ax[0][0].set_xticklabels(tmp_m["ts"].to_numpy()[range(0, len(tmp_m["ts"]), 12)], rotation=45)
            ax[0][0].set_title(f"Monthly t-stats of ic_{c}.")

            tmp_m_n = m.filter(pl.col("var") == f"{c}_neutralized").sort("ts")
            
            ax[0][1].bar(
                x = tmp_m_n["ts"],
                height = tmp_m_n["t-stats"]
            )
            ax[0][1].set_xticks(range(0, len(tmp_m_n["ts"]), 12))
            ax[0][1].set_xticklabels(tmp_m_n["ts"].to_numpy()[range(0, len(tmp_m["ts"]), 12)], rotation=45)
            ax[0][1].set_title(f"Monthly t-stats of ic_{c}_neutralized.")
            
            tmp_y = y.filter(pl.col("var") == c).sort("year")

            ax[1][0].bar(
                x = tmp_y["year"],
                height = tmp_y["t-stats"]
            )
            ax[1][0].set_title(f"Yearly t-stats of ic_{c}.")

            tmp_y_n = y.filter(pl.col("var") == f"{c}_neutralized").sort("year")
    
            ax[1][1].bar(
                x = tmp_y_n["year"],
                height = tmp_y_n["t-stats"]
            )
            ax[1][1].set_title(f"Yearly t-stats of ic_{c}_neutralized.")

            ax[2][0].bar(
                x = tmp_m["ts"],
                height = tmp_m["ir"]
            )
            ax[2][0].set_xticks(range(0, len(tmp_m["ts"]), 12))
            ax[2][0].set_xticklabels(tmp_m["ts"].to_numpy()[range(0, len(tmp_m["ts"]), 12)], rotation=45)
            ax[2][0].set_title(f"Monthly ir of ic_{c}.")
            
            ax[2][1].bar(
                x = tmp_m_n["ts"],
                height = tmp_m_n["ir"]
            )
            ax[2][1].set_xticks(range(0, len(tmp_m_n["ts"]), 12))
            ax[2][1].set_xticklabels(tmp_m_n["ts"].to_numpy()[range(0, len(tmp_m["ts"]), 12)], rotation=45)
            ax[2][1].set_title(f"Monthly ir of ic_{c}_neutralized.")

            ax[3][0].bar(
                x = tmp_y["year"],
                height = tmp_y["ir"]
            )
            ax[3][0].set_title(f"Yearly ir of ic_{c}.")
    
            ax[3][1].bar(
                x = tmp_y["year"],
                height = tmp_y["ir"]
            )
            ax[3][1].set_title(f"Yearly ir of ic_{c}_neutralized.")

            ic1 = ic_df.sort(["year", "month", "day"])[c].to_numpy().flatten()
            ic2 = ic_df.sort(["year", "month", "day"])[f"{c}_neutralized"].to_numpy().flatten()
            if ic1.std() == 0:
                ir1 = 0
            else:
                ir1 = ic1.mean() / ic1.std()
            if ic2.std() == 0:
                ir2 = 0
            else:
                ir2 = ic2.mean() / ic2.std()


            ax[4].plot(ic_df["time"], np.cumsum(ic1), label=c)
            ax[4].plot(ic_df["time"], np.cumsum(ic2), label=f"{c}_neutralized")
            ax[4].set_title(f"Cumulative ic of {c} and {c}_neutralized")
            ax[4].legend()

            plt.tight_layout()
            plt.show()
            print(f"Daily IC/IR of {c}:")
            print(f"Raw: IC: {ic1.mean()}. IR: {ir1}.")
            print(f"Neutralized: IC: {ic2.mean()}. IR: {ir2}.")
    elif need_figure:
        for c in features:
            if c in to_skip:
                continue
            fig = plt.figure(figsize=(24,24))
            gs = fig.add_gridspec(5,1)
            ax = [[None, None] for _ in range(5)]
            for i in range(5):
                if i == 4:
                    continue
                for j in range(1):
                    ax[i][j] = fig.add_subplot(gs[i,j])
            ax[4] = fig.add_subplot(gs[4,:])
            
            tmp_m = m.filter(pl.col("var") == c).sort("ts")
            
            ax[0][0].bar(
                x = tmp_m["ts"],
                height = tmp_m["t-stats"]
            )
            ax[0][0].set_xticks(range(0, len(tmp_m["ts"]), 12))
            ax[0][0].set_xticklabels(tmp_m["ts"].to_numpy()[range(0, len(tmp_m["ts"]), 12)], rotation=45)
            ax[0][0].set_title(f"Monthly t-stats of ic_{c}.")
            
            tmp_y = y.filter(pl.col("var") == c).sort("year")

            ax[1][0].bar(
                x = tmp_y["year"],
                height = tmp_y["t-stats"]
            )
            ax[1][0].set_title(f"Yearly t-stats of ic_{c}.")

            ax[2][0].bar(
                x = tmp_m["ts"],
                height = tmp_m["ir"]
            )
            ax[2][0].set_xticks(range(0, len(tmp_m["ts"]), 12))
            ax[2][0].set_xticklabels(tmp_m["ts"].to_numpy()[range(0, len(tmp_m["ts"]), 12)], rotation=45)
            ax[2][0].set_title(f"Monthly ir of ic_{c}.")

            ax[3][0].bar(
                x = tmp_y["year"],
                height = tmp_y["ir"]
            )
            ax[3][0].set_title(f"Yearly ir of ic_{c}.")

            ic1 = ic_df.sort(["year", "month", "day"])[c].to_numpy().flatten()
            if ic1.std() == 0:
                ir1 = 0
            else:
                ir1 = ic1.mean() / ic1.std()

            ax[4].plot(ic_df["time"], np.cumsum(ic1), label=c)
            ax[4].set_title(f"Cumulative ic of {c}")
            ax[4].legend()

            plt.tight_layout()
            plt.show()
            print(f"Daily IC/IR of {c}:")
            print(f"IC: {ic1.mean()}. IR: {ir1}.")
    return m, y, ir_dict, selected

def feature_test_symbol_wise(
    df: pl.LazyFrame | pl.DataFrame, 
    features=[], 
    target: str="ret_30min",
    t_th: float=2.5,
    need_figure: bool=True,
):
    if isinstance(df, pl.LazyFrame):
        cur_cols = df.collect_schema().names()
    elif isinstance(df, pl.DataFrame):
        cur_cols = df.columns

    if len(features) == 0:
        features = [c for c in cur_cols if "feature" in c]

    df = df.filter((pl.col("Time").dt.year() < 2021) & (pl.col("Time").dt.year() > 2013))
    
    df = df.with_columns([
        pl.col("Time").dt.year().alias("year"),
        pl.col("Time").dt.month().alias("month"),
        pl.col("Time").dt.day().alias("day")
    ]).fill_nan(None).fill_null(0)

    ic_df = df.group_by(["year", "month", "day", "Symbol"]).agg([
        pl.corr(
            pl.col(c),
            pl.col(target)
        ).alias(c)
        for c in features
    ]).fill_nan(None).fill_null(0)

    if isinstance(ic_df, pl.LazyFrame):
        ic_df = ic_df.collect()
        
    ic_df = ic_df.with_columns([
        (
            pl.col("year").cast(pl.String) + "-" + 
             pl.col("month").cast(pl.String).str.zfill(2) + "-" +
             pl.col("day").cast(pl.String).str.zfill(2)
        ).str.strptime(pl.Date, "%Y-%m-%d").alias("time")
    ])
    
    ic_df = ic_df.sort("time")
    long = ic_df.drop("time").unpivot(
        index = ["year", "month", "day", "Symbol"],
        variable_name = "var",
        value_name = "value",
    )

    y = long.group_by(["year", "var", "Symbol"]).agg([
        pl.len().alias("T"),
        pl.col("value").mean().alias("ic_mean"),
        (
            pl.col("value").mean() /
            (pl.col("value").std() + 1e-12)
        ).alias("ir"),
        (
            pl.col("value").mean() /
            (
                (pl.col("value").map_batches(
                    lambda arr: NW_var(arr),
                    return_dtype=pl.Float64,
                    returns_scalar=True
                ) / (pl.len() + 1e-12)).sqrt() + 1e-12
            )
        ).alias("t-stats"),
    ]).with_columns([
        pl.when(
            (pl.col("t-stats").is_nan()) | (pl.col("t-stats").is_null())
        ).then(
            pl.col("ir") * pl.col("T").sqrt()
        ).otherwise(pl.col("t-stats")).alias("t-stats")
    ])

    print(f"Selecting features with t-stats > {t_th}...")

    y_agg = y.group_by(["var", "year"]).agg([
        (pl.col("t-stats") > t_th).sum().alias("n_pos"),
        (pl.col("t-stats") < -t_th).sum().alias("n_neg"),
        (pl.col("t-stats").abs() < t_th).sum().alias("n_ns"),
        pl.len().alias("n_total"),
        pl.col("t-stats")
    ]).sort(["var", "year"])
    
    selected = y_agg.group_by("var").agg([
        (pl.col("n_ns") / pl.col("n_total")).quantile(0.9).alias("ns"),
        (pl.col("n_ns") / pl.col("n_total")).max().alias("max_ns"),
    ]).filter(pl.col("ns") < 0.2)["var"].to_list()
    
    ir_df = ic_df.filter(
        pl.col("time").dt.year() < 2018
    ).select([
        (pl.col(c).mean() / (pl.col(c).std() + 1e-12)).alias(c)
        for c in features
    ]).unpivot().rename({"variable": "feature", "value": "ir"})

    ir_dict = dict(zip(ir_df["feature"], ir_df["ir"]))

    if need_figure:
        for c in features:
            y_s = y.filter(pl.col("var") == c).filter(pl.col("year") <= 2020)
            grouped = []
            years = []
            for yr, g in y_s.sort("year").group_by("year"):
                years.append(yr)
                grouped.append(g["t-stats"])
            
            x = np.arange(len(years))
            
            cmap = plt.get_cmap("tab20")
            color_map = {s: cmap(i % cmap.N) for i, s in enumerate(SYMBOLS)}
            
            fig, ax = plt.subplots(figsize=(8,4))
            _ = ax.violinplot(
                grouped,
                positions=x,
                widths=0.6,
                showmeans=True,
                showextrema=True,
            )
            max_y = -0.1
            for i, yr in enumerate(years):
                vals = y_s.filter(pl.col("year") == yr[0])["t-stats"]
                syms = y_s.filter(pl.col("year") == yr[0])["Symbol"].unique().to_list()
        
                n_pos = (vals > t_th).sum()
                n_neg = (vals < -t_th).sum()
                n_not_eff = (vals.abs() <= t_th).sum()
                
                jitter_x = np.random.uniform(-0.15, 0.15, size=len(vals))
                colors = [color_map[s] for s in syms]
                ax.scatter(
                    x[i] + jitter_x,
                    vals,
                    s=10,
                    alpha=0.5,
                    c=colors
                )
                y_top = float(vals.max()) if len(vals) > 0 else 0.0
                max_y = max(max_y, y_top)
                ax.text(
                    x[i],
                    y_top + 0.3,
                    f"+{n_pos} / {n_not_eff} / -{n_neg}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, max_y + 2)
            ax.set_xticks(x)
            ax.set_xticklabels(years)
            ax.set_xlabel("Year")
            ax.set_title(f"t-stats of {c} by year & symbol")
            plt.tight_layout()
            plt.show()
    return y, ir_dict, selected
                
def feature_selection(
    df: pl.LazyFrame | pl.DataFrame,
    features=None,
    ir_dict=None,
    corr_th: float=0.7,
    need_figure: bool=False,
):
    print("Start clustering...")
    last_num = -1

    if isinstance(df, pl.LazyFrame):
        if features is None:
            features = [c for c in df.collect_schema().names() if "feature" in c]
        df = df.select(features).collect()
    elif isinstance(df, pl.DataFrame):
        if features is None:
            features = [c for c in df.columns if "feature" in c]
    
    tmp = features
    for _ in range(10):
        if len(tmp) <= 1:
            break
        corr = df.select(tmp).to_pandas().corr()
        dist = 1 - corr.abs().values
        np.fill_diagonal(dist, 0.0)
        dist = dist * (dist > 0).astype(float)
        condensed = squareform(dist, checks=False)
        Z = fastcluster.linkage(condensed, method="average")
        labels = fcluster(Z, t=1-corr_th, criterion="distance")
        from collections import defaultdict
        cluster = defaultdict(list)
        for f, lab in zip(tmp, labels):
            cluster[int(lab)].append(f)

        tmp = []
        for g in cluster.values():
            if len(g) == 1:
                tmp += g
                continue
            tmp.append(max(g, key=lambda k: abs(ir_dict[k])))
            
        tmp = list(set(tmp))
        last_num = len(tmp)
        if need_figure:
            plt.figure(figsize=(5,2))
            dendrogram(
                Z,
                labels=corr.columns,
                leaf_rotation=90,
            )
            plt.show()
            heatmap = sns.clustermap(
                corr,
                method="average",
                metric="correlation",
                cmap="vlag",
                center=0,
                figsize=(8,8)
            )
            plt.tight_layout()
            plt.show()
        if len(tmp) == last_num:
            break
    print(f"Now we have {len(tmp)} features...")

    return tmp
    