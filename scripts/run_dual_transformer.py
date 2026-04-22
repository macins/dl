# scripts/run_exp/run_dual_transformer.py

import os
import sys
from datetime import datetime, timedelta
import argparse
import random

import numpy as np
import polars as pl
import torch

# ====== 路径设置：假定脚本在项目根目录的子目录下 ======
PROJ_ROOT = os.getcwd().rsplit("/", 1)[0]
if PROJ_ROOT not in sys.path:
    sys.path.append(PROJ_ROOT)

from src.preprocessor import CustomScaler
from dl.trainer import TrainConfig, Trainer
from dl.pipeline import RollingTrainingPipeline

# 如果你的 DualAxisTransformer 放的位置不一样，把这行改掉即可
from dl.models.M4 import DualAxisTransformer


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run DualAxisTransformer experiment.")

    # ===== loss 权重 =====
    parser.add_argument("--w-mse", type=float, default=1.0,
                        help="Weight for MSE loss.")
    parser.add_argument("--w-corr", type=float, default=0.0,
                        help="Weight for Corr loss.")
    parser.add_argument("--w-ret", type=float, default=0.0,
                        help="Weight for Ret loss.")
    parser.add_argument("--w-custom", type=float, default=0.0,
                        help="Weight for Custom loss.")
    parser.add_argument("--w-custom-mse", type=float, default=0.0,
                        help="Weight for Custom MSE loss.")

    # ===== 优化相关 =====
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate for AdamW.")
    parser.add_argument("--weight-decay", type=float, default=3e-4,
                        help="Weight decay for AdamW.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs per window.")
    parser.add_argument("--lookback", type=int, default=120,
                        help="Lookback length used in CustomDataset.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")

    # ===== 是否使用类别特征 / 静态线性 =====
    parser.add_argument("--use-cat-emb", type=int, default=1, choices=[0, 1],
                        help="Whether to use categorical embeddings.")
    parser.add_argument("--use-static-linear", type=int, default=1, choices=[0, 1],
                        help="Whether to use static linear baseline on last step.")

    # ===== 时间分支结构 =====
    parser.add_argument("--use-time-branch", type=int, default=1, choices=[0, 1],
                        help="Whether to enable time branch.")
    parser.add_argument("--d-model-time", type=int, default=128,
                        help="Time branch d_model.")
    parser.add_argument("--nhead-time", type=int, default=4,
                        help="Time branch num heads.")
    parser.add_argument("--num-layers-time", type=int, default=1,
                        help="Time branch num Transformer layers.")
    parser.add_argument("--dim-ff-time", type=int, default=128,
                        help="Time branch FFN hidden dim.")
    parser.add_argument("--dropout-time", type=float, default=0.1,
                        help="Time branch dropout.")
    parser.add_argument("--use-rope-time", type=int, default=0, choices=[0, 1],
                        help="Whether to use RoPE in time branch.")
    parser.add_argument("--time-is-causal", type=int, default=0, choices=[0, 1],
                        help="Whether time attention is causal.")
    parser.add_argument("--use-fox-time", type=int, default=0, choices=[0, 1],
                        help="Whether to use FoX gate in time branch.")
    parser.add_argument("--time-pool", type=str, default="last",
                        choices=["last", "mean", "cls"],
                        help="Pooling type for time branch.")

    # ===== 特征分支结构 =====
    parser.add_argument("--use-feat-branch", type=int, default=1, choices=[0, 1],
                        help="Whether to enable feature branch.")
    parser.add_argument("--d-model-feat", type=int, default=256,
                        help="Feature branch d_model.")
    parser.add_argument("--nhead-feat", type=int, default=8,
                        help="Feature branch num heads.")
    parser.add_argument("--num-layers-feat", type=int, default=1,
                        help="Feature branch num Transformer layers.")
    parser.add_argument("--dim-ff-feat", type=int, default=128,
                        help="Feature branch FFN hidden dim.")
    parser.add_argument("--dropout-feat", type=float, default=0.1,
                        help="Feature branch dropout.")
    parser.add_argument("--feat-pool", type=str, default="mean",
                        choices=["mean", "cls"],
                        help="Pooling type for feature branch.")
    parser.add_argument("--feat-history-len", type=int, default=1,
                        help="How many recent timesteps to aggregate for feature branch.")
    parser.add_argument("--feat-history-agg", type=str, default="learned",
                        choices=["mean", "learned"],
                        help="Temporal aggregation for feature branch base features.")

    # ===== Head 结构 =====
    parser.add_argument("--head-hidden", type=int, default=128,
                        help="Hidden dim of final MLP head.")

    # ===== time patching =====
    parser.add_argument("--use-time-patch", type=int, default=1, choices=[0, 1],
                        help="Whether to enable time patching.")
    parser.add_argument("--time-patch-len", type=int, default=10,
                        help="Time patch length (number of timesteps per patch).")
    parser.add_argument("--time-patch-stride", type=int, default=10,
                        help="Time patch stride.")

    # ===== rolling 窗口设置 =====
    parser.add_argument("--start-date", type=str, default="2015-01-01",
                        help="Rolling start date, e.g. 2015-01-01")
    parser.add_argument("--end-date", type=str, default="2021-07-01",
                        help="Rolling end date, e.g. 2021-07-01")
    parser.add_argument("--slice-days", type=int, default=365,
                        help="Length of each rolling slice in days.")

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    set_global_seed(args.seed)

    # ====== 数据列设置 ======
    categorical_cols = ["Symbol", "DATA_sector"]
    time_col = "Time"
    target_col = "ret_30min"
    symbol_col = "Symbol"
    base_col = None
    misc_cols = ["DATA_is_trading_hour", "DATA_is_last_30min"]

    # ====== 读数据（LazyFrame） ======
    df = pl.scan_parquet("data/cache/data.parquet")
    feature_cols = [c for c in df.collect_schema().names() if "feature" in c]

    # 只保留需要的列
    all_cols = [time_col, target_col] + feature_cols + categorical_cols + misc_cols
    df = df.select(all_cols)

    # ====== 类别特征基数 ======
    if args.use_cat_emb:
        cat_cardinals = [52, 7]
    else:
        cat_cardinals = []

    # ====== 构建 DualAxisTransformer 模型 ======
    model = DualAxisTransformer(
        num_numeric_features=len(feature_cols),
        cat_cardinals=cat_cardinals,
        # time branch
        d_model_time=args.d_model_time,
        nhead_time=args.nhead_time,
        num_layers_time=args.num_layers_time,
        dim_ff_time=args.dim_ff_time,
        dropout_time=args.dropout_time,
        use_fox_time=bool(args.use_fox_time),
        use_rope_time=bool(args.use_rope_time),
        time_is_causal=bool(args.time_is_causal),
        # feature branch
        d_model_feat=args.d_model_feat,
        nhead_feat=args.nhead_feat,
        num_layers_feat=args.num_layers_feat,
        dim_ff_feat=args.dim_ff_feat,
        dropout_feat=args.dropout_feat,
        feat_history_len=args.feat_history_len,
        feat_history_agg=args.feat_history_agg,
        # head & switches
        head_hidden=args.head_hidden,
        use_static_linear=bool(args.use_static_linear),
        use_time_branch=bool(args.use_time_branch),
        use_feat_branch=bool(args.use_feat_branch),
        time_pool=args.time_pool,
        feat_pool=args.feat_pool,
        # time patch
        use_time_patch=bool(args.use_time_patch),
        time_patch_len=args.time_patch_len,
        time_patch_stride=args.time_patch_stride,
    )

    # ====== loss_weights 配置 ======
    loss_weights = {
        "mse": args.w_mse,
        "corr": args.w_corr,
        "ret": args.w_ret,
        "custom": args.w_custom,
        "custom_mse": args.w_custom_mse,
    }

    # ====== 实验名：尽量把结构与超参都 encode 进去 ======
    exp_name = (
        "dualtr"
        f"_lb{args.lookback}"
        f"_dt{args.d_model_time}_nt{args.nhead_time}_lt{args.num_layers_time}"
        f"_dfft{args.dim_ff_time}_drt{args.dropout_time:g}"
        f"_df{args.d_model_feat}_nf{args.nhead_feat}_lf{args.num_layers_feat}"
        f"_dfff{args.dim_ff_feat}_drf{args.dropout_feat:g}"
        f"_fhl{args.feat_history_len}_fha{args.feat_history_agg}"
        f"_tpool{args.time_pool}_fpool{args.feat_pool}"
        f"_rope{args.use_rope_time}_causal{args.time_is_causal}_fox{args.use_fox_time}"
        f"_tp{args.use_time_patch}_tpl{args.time_patch_len}_tps{args.time_patch_stride}"
        f"_cat{args.use_cat_emb}_sl{args.use_static_linear}"
        f"_tb{args.use_time_branch}_fb{args.use_feat_branch}"
        f"_lr{args.lr:g}_wd{args.weight_decay:g}"
        f"_wm{args.w_mse:g}_wc{args.w_corr:g}_wr{args.w_ret:g}"
        f"_wcu{args.w_custom:g}_wcum{args.w_custom_mse:g}"
        f"_seed{args.seed}"
    )

    # ====== TrainConfig ======
    cfg = TrainConfig(
        epochs=args.epochs,
        ckpt_dir=os.path.join("ckpt", exp_name),
        log_root="exp_log",
        exp_name=exp_name,
        opt_type="AdamW",
        lr=args.lr,
        weight_decay=args.weight_decay,
        sched_type=None,        
        T_max=20,
        min_lr=5e-7,
        use_amp=True,
        grad_accum=1,
        max_grad_norm=1.0,
        patience=0,
        loss_weights=loss_weights,
        log_param_hist=False,
        model_name="DualAxisTransformer",
        model_hparams={
            "num_numeric_features": len(feature_cols),
            "cat_cardinals": cat_cardinals,
            "d_model_time": args.d_model_time,
            "nhead_time": args.nhead_time,
            "num_layers_time": args.num_layers_time,
            "dim_ff_time": args.dim_ff_time,
            "dropout_time": args.dropout_time,
            "use_rope_time": bool(args.use_rope_time),
            "time_is_causal": bool(args.time_is_causal),
            "use_fox_time": bool(args.use_fox_time),
            "d_model_feat": args.d_model_feat,
            "nhead_feat": args.nhead_feat,
            "num_layers_feat": args.num_layers_feat,
            "dim_ff_feat": args.dim_ff_feat,
            "dropout_feat": args.dropout_feat,
            "feat_history_len": args.feat_history_len,
            "feat_history_agg": args.feat_history_agg,
            "time_pool": args.time_pool,
            "feat_pool": args.feat_pool,
            "use_static_linear": bool(args.use_static_linear),
            "use_time_branch": bool(args.use_time_branch),
            "use_feat_branch": bool(args.use_feat_branch),
            "lookback": args.lookback,
            "use_time_patch": bool(args.use_time_patch),
            "time_patch_len": args.time_patch_len,
            "time_patch_stride": args.time_patch_stride,
        },
    )

    trainer = Trainer(model, cfg)

    # ====== Rolling pipeline ======
    save_dir = os.path.join("results", exp_name)

    pipeline = RollingTrainingPipeline(
        model=model,
        trainer=trainer,
        lazy_df=df,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        preprocessor=CustomScaler(scale=True),
        clip_train=True,
        time_col=time_col,
        symbol_col=symbol_col,
        target_col=target_col,
        base_col=base_col,
        lookback=args.lookback,
        save_dir=save_dir,
        train_batch_size=4096,
        val_batch_size=4096,
        test_batch_size=4096,
    )

    start_time = datetime.fromisoformat(args.start_date)
    end_time = datetime.fromisoformat(args.end_date)
    slice_len = timedelta(days=args.slice_days)

    _ = pipeline.run(start_time, end_time, slice_len, save_pred=True)

    print(f"Experiment {exp_name} finished. Results saved under {save_dir}.")


if __name__ == "__main__":
    main()
