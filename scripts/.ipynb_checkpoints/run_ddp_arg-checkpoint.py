import os
import sys
import random
import argparse
from datetime import datetime

import numpy as np
import polars as pl

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

PROJ_ROOT = os.getcwd().rsplit("/", 1)[0]
sys.path.append(PROJ_ROOT)

from src.backtest import analyze_signal  # noqa: F401
from src.preprocessor import CustomScaler

from dl.dataio import (
    CustomDataset,
    custom_collate_fn,
    CrossSectionDataset,
    cross_section_collate_fn,
)
from dl.models.gnn import SymbolSectorBipartiteGNN
from dl.metrics import CorrLoss, RetLoss, CustomLoss
from dl.trainer import TrainConfig, Trainer, DDPTrainer

from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import gc


# ============================================================
# 一些小工具
# ============================================================

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_abbr(s: str) -> str:
    words = s.split("_")
    return "".join([w[0] for w in words])


def _pl_date(dt: datetime):
    """把 datetime 转成 polars 里的 datetime expr"""
    return pl.datetime(dt.year, dt.month, dt.day)


# ============================================================
# DDP 训练主逻辑（函数）
# ============================================================

def run_ddp(
    train_start: datetime,
    train_end: datetime,
    val_start: datetime,
    val_end: datetime,
    model_params: dict,
    cfg: TrainConfig,
    lookback: int = 30,
    batch_size: int = 512,
):
    """
    单机多卡 DDP 训练入口：
    - 假设外部已经 dist.init_process_group(...)
    - 使用 CrossSectionDataset + SymbolSectorBipartiteGNN + DDPTrainer
    """

    assert dist.is_initialized(), "请先在外部调用 dist.init_process_group 再调用 run_ddp()"

    # --------- DDP rank / world_size / local_rank ---------
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    torch.cuda.set_device(local_rank)

    # --------- 时间段转成 polars filter ---------
    period_tr = (pl.col("Time") >= _pl_date(train_start)) & (pl.col("Time") < _pl_date(train_end))
    period_va = (pl.col("Time") >= _pl_date(val_start)) & (pl.col("Time") < _pl_date(val_end))

    # --------- 列定义 ---------
    categorical_cols = ["Symbol", "DATA_sector", "DATA_month", "DATA_is_day_session"]
    cat_card_dict = {
        "Symbol": 52,
        "DATA_sector": 7,
        "DATA_month": 13,
        "DATA_weekday": 6,
        "DATA_is_day_session": 3,
    }
    time_col = "Time"
    target_col = "ret_30min"
    base_col = None
    misc = ["Symbol", "DATA_is_trading_hour", "DATA_is_last_30min", "DATA_is_bad_data"]

    # --------- 读 parquet + 选列 ---------
    df_lazy = pl.scan_parquet("../data/cache/data.parquet").filter(period_tr | period_va)
    feature_cols = [
        c for c in df_lazy.collect_schema().names() if "feature" in c
    ] + ["OpenInterest", "LogVolume", "LogAmount"]

    df = df_lazy.select(
        list(set([time_col, target_col] + feature_cols + categorical_cols + misc))
    ).collect()

    # --------- 标准化（和你原来一样） ---------
    my_scaler = CustomScaler(features=feature_cols, scale=True)
    _ = my_scaler.fit_transform(df.filter(period_tr & (~pl.col("DATA_is_bad_data"))))
    df = my_scaler.transform(df)

    # --------- 构造模型参数（填 num_numeric_features / cat_cardinals） ---------
    model_params = dict(model_params)  # 拷一份，别改原来的
    model_params["num_numeric_features"] = len(feature_cols)
    model_params["cat_cardinals"] = [cat_card_dict[k] for k in categorical_cols]
    model_params.setdefault("num_sectors", 7)

    model = SymbolSectorBipartiteGNN(**model_params)

    # --------- Dataset & DataLoader ---------
    T = lookback
    train_set = CrossSectionDataset(
        df=df.filter(period_tr & (~pl.col("DATA_is_bad_data"))),
        feature_cols=feature_cols,
        target_col=target_col,
        categorical_cols=categorical_cols,
        lookback=T,
        base_col=base_col,
        need_clip=True,
    )
    val_set = CrossSectionDataset(
        df=df.filter(period_va),
        feature_cols=feature_cols,
        target_col=target_col,
        categorical_cols=categorical_cols,
        lookback=T,
        base_col=base_col,
    )

    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_set, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=cross_section_collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=cross_section_collate_fn,
    )

    # --------- 补齐 cfg 的一些字段 ---------
    if cfg.loss_weights is None:
        cfg.loss_weights = {
            "mse": 1.0,
            "corr": 0.0,
            "custom": 0.0,
            "ret": 0.0,
            "custom_mse": 0.0,
            "sharpe": 0.0,
        }

    cfg.model_name = cfg.model_name or "CrossSectionTimeModel"
    cfg.model_hparams = model_params

    trainer = DDPTrainer(model, cfg, rank=rank, world_size=world_size, local_rank=local_rank)
    trainer.fit(train_loader, val_loader)

    return trainer


# ============================================================
# argparse：命令行参数
# ============================================================

def parse_int_list(s: str):
    if not s:
        return []
    return [int(x) for x in s.split(",")]

def parse_args():
    parser = argparse.ArgumentParser(description="DDP training for CrossSection GNN+Time model")

    # --- 时间段 ---
    parser.add_argument("--train-start", type=str, default="2014-01-01")
    parser.add_argument("--train-end", type=str, default="2018-01-01")
    parser.add_argument("--val-start", type=str, default="2018-01-01")
    parser.add_argument("--val-end", type=str, default="2019-01-01")

    # --- 通用训练超参 ---
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max-grad-norm", type=float, default=1e-3)

    parser.add_argument("--sched-type", type=str, default="step",
                        choices=["none", "warm", "cosine", "step"])
    parser.add_argument("--T-max", type=int, default=20)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--milestones", type=parse_int_list, default="5,10,15,25,35,45")

    parser.add_argument("--exp-name", type=str, default="")
    parser.add_argument("--ckpt-root", type=str, default="../ddp/ckpt")
    parser.add_argument("--log-root", type=str, default="../ddp/exp_log")

    # --- Time branch 超参 ---
    parser.add_argument("--d-model-time", type=int, default=256)
    parser.add_argument("--nhead-time", type=int, default=8)
    parser.add_argument("--num-layers-time", type=int, default=1)
    parser.add_argument("--dim-ff-time", type=int, default=384)
    parser.add_argument("--dropout-time", type=float, default=0.2)

    parser.add_argument("--ffn-gate-type-time", type=str, default="glu",
                        choices=["none", "gelu", "glu", "geglu", "swiglu"])
    parser.add_argument("--attn-gate-type-time", type=str, default="element",
                        choices=["none", "head", "element"])

    parser.add_argument("--time-pool", type=str, default="last",
                        choices=["last", "mean", "max", "none"])

    parser.add_argument("--use-rope-time", action="store_true", default=True)
    parser.add_argument("--no-rope-time", action="store_false", dest="use_rope_time")

    parser.add_argument("--time-is-causal", action="store_true", default=True)
    parser.add_argument("--noncausal-time", action="store_false", dest="time_is_causal")

    # --- GNN / sector 部分 ---
    parser.add_argument("--num-gnn-layers", type=int, default=0)
    parser.add_argument("--gnn-dim-ff-symbol", type=int, default=256)
    parser.add_argument("--gnn-dim-ff-sector", type=int, default=384)

    parser.add_argument("--gnn-ffn-gate-symbol", type=str, default="glu",
                        choices=["none", "gelu", "glu", "geglu", "swiglu"])
    parser.add_argument("--gnn-ffn-gate-sector", type=str, default="glu",
                        choices=["none", "gelu", "glu", "geglu", "swiglu"])

    parser.add_argument("--use-sector-attn", action="store_true", default=True)
    parser.add_argument("--no-sector-attn", action="store_false", dest="use_sector_attn")

    parser.add_argument("--sector-nhead", type=int, default=8)
    parser.add_argument("--sector-attn-dropout", type=float, default=0.2)
    parser.add_argument("--sector-attn-gate-type", type=str, default="element",
                        choices=["none", "head", "element"])

    parser.add_argument("--gnn-agg-mode", type=str, default="attn",
                        choices=["attn", "mean", "sum", "max"])

    parser.add_argument("--use-learned-cluster", action="store_true", default=True)
    parser.add_argument("--no-learned-cluster", action="store_false", dest="use_learned_cluster")
    parser.add_argument("--num-learned-clusters", type=int, default=2)
    parser.add_argument("--cluster-topk", type=int, default=None)
    parser.add_argument("--sym-topk", type=int, default=None)
    parser.add_argument("--use-moe", action="store_true", default=True)
    parser.add_argument("--no-moe", action="store_false", dest="use_moe")
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--moe-top-k", type=int, default=2)

    args = parser.parse_args()
    return args


# ============================================================
# main：解析参数 + init DDP + 调 run_ddp
# ============================================================

def main():
    set_global_seed()
    dist.init_process_group(backend="nccl")

    args = parse_args()

    # ---- 解析时间 ----
    train_start = datetime.fromisoformat(args.train_start)
    train_end = datetime.fromisoformat(args.train_end)
    val_start = datetime.fromisoformat(args.val_start)
    val_end = datetime.fromisoformat(args.val_end)

    # ---- 构造 model_params ----
    model_params = {
        "num_numeric_features": None,   # run_ddp 里填
        "cat_cardinals": None,         # run_ddp 里填
        "num_sectors": 7,
        "d_model_time": args.d_model_time,
        "nhead_time": args.nhead_time,
        "num_layers_time": args.num_layers_time,
        "dim_ff_time": args.dim_ff_time,
        "dropout_time": args.dropout_time,
        "use_rope_time": args.use_rope_time,
        "time_is_causal": args.time_is_causal,
        "time_pool": args.time_pool,
        "ffn_gate_type_time": args.ffn_gate_type_time,
        "attn_gate_type_time": args.attn_gate_type_time,
        "num_gnn_layers": args.num_gnn_layers,
        "gnn_dim_ff_symbol": args.gnn_dim_ff_symbol,
        "gnn_ffn_gate_symbol": args.gnn_ffn_gate_symbol,
        "gnn_dim_ff_sector": args.gnn_dim_ff_sector,
        "gnn_ffn_gate_sector": args.gnn_ffn_gate_sector,
        "use_sector_attn": args.use_sector_attn,
        "sector_nhead": args.sector_nhead,
        "sector_attn_dropout": args.sector_attn_dropout,
        "sector_attn_gate_type": args.sector_attn_gate_type,
        "gnn_agg_mode": args.gnn_agg_mode,
        "use_learned_cluster": args.use_learned_cluster,
        "num_learned_clusters": args.num_learned_clusters,
        "cluster_topk": args.cluster_topk,
        "sym_topk": args.sym_topk,
        "use_moe": args.use_moe,
        "num_experts": args.num_experts,
        "moe_top_k": args.moe_top_k,
    }

    # ---- 如果没给 exp_name，就自动拼一个 ----
    if args.exp_name:
        exp_name = args.exp_name
    else:
        exp_name = f"cstr|lb={args.lookback}|"
        for k, v in model_params.items():
            if k in ["num_numeric_features", "cat_cardinals"]:
                continue
            exp_name += f"{get_abbr(k)}={v}|"

    # ---- 构造 TrainConfig ----
    loss_weights = {
        "mse": 1.0,
        "corr": 0.0,
        "custom": 0.0,
        "ret": 0.0,
        "custom_mse": 0.0,
        "sharpe": 0.0,
    }

    if args.sched_type == "none":
        sched_type = None
    else:
        sched_type = args.sched_type

    cfg = TrainConfig(
        epochs=args.epochs,
        ckpt_dir=os.path.join(args.ckpt_root, exp_name),
        log_root=args.log_root,
        exp_name=exp_name,
        opt_type="AdamW",
        lr=args.lr,
        sched_type=sched_type,
        T_max=args.T_max,
        min_lr=args.min_lr,
        gamma=args.gamma,
        milestones=args.milestones,
        use_amp=True,
        grad_accum=1,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        patience=args.patience,
        loss_weights=loss_weights,
        log_param_hist=False,
        model_name="CrossSectionTimeModel",
        model_hparams=model_params,
    )

    # ---- 真正跑 DDP 训练 ----
    run_ddp(
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        model_params=model_params,
        cfg=cfg,
        lookback=args.lookback,
        batch_size=args.batch_size,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    # 例如：
    # torchrun --nproc_per_node=4 run_ddp.py \
    #   --train-start 2014-01-01 --train-end 2018-01-01 \
    #   --val-start 2018-01-01 --val-end 2019-01-01 \
    #   --lr 1e-4 --epochs 200 --d-model-time 256
    main()
