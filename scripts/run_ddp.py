import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

import sys
PROJ_ROOT = os.getcwd().rsplit("/",1)[0]
sys.path.append(PROJ_ROOT)

from src.utils import get_ret, get_data, get_training_data, get_corr
from config import DataConfig, get_session_expr
data_cfg = DataConfig()
SYMBOLS = data_cfg.SYMBOLS

from datetime import datetime, timedelta
import numpy as np
import polars as pl
from src.backtest import analyze_signal
from src.preprocessor import CustomScaler
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
import gc

from sklearn.metrics import r2_score

from dl.dataio import CustomDataset, custom_collate_fn, CrossSectionDataset, cross_section_collate_fn
from dl.models.cs import CrossSectionTimeModel
from dl.models.sp import SpatioTemporalGridModel
from dl.models.gnn import SymbolSectorBipartiteGNN, GNNThenTimeTransformer
from dl.metrics import CorrLoss, RetLoss, CustomLoss
from dl.trainer import TrainConfig, Trainer, DDPTrainer
from torch.utils.data import DataLoader
from torch.nn import MSELoss, L1Loss

import torch
import random

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def get_abbr(s: str):
    words = s.split("_")
    return "".join([w[0] for w in words])
    
def main():
    set_global_seed()
    dist.init_process_group(backend="nccl")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    torch.cuda.set_device(local_rank)

    # dataset & dataloader
    period_tr = (pl.col("Time") < pl.datetime(2017,1,1)) & (pl.col("Time") >= pl.datetime(2014,1,1))
    period_va = (pl.col("Time") < pl.datetime(2018,1,1)) & (pl.col("Time") >= pl.datetime(2017,1,1))
    period_te1 = (pl.col("Time") < pl.datetime(2019,1,1)) & (pl.col("Time") >= pl.datetime(2018,1,1))
    period_te2 = (pl.col("Time") < pl.datetime(2020,1,1)) & (pl.col("Time") >= pl.datetime(2019,1,1))
    period_te3 = pl.col("Time") >= pl.datetime(2020,1,1)
    
    categorical_cols = ['Symbol', 'DATA_sector', 'DATA_month', 'DATA_is_day_session']
    cat_card_dict = {"Symbol": 52, "DATA_sector": 7, "DATA_month": 13, "DATA_weekday": 6, "DATA_is_day_session": 3}
    time_col = "Time"
    target_col = "ret_30min"
    base_col = None
    misc = ["Symbol", "DATA_is_trading_hour", "DATA_is_last_30min", "DATA_is_bad_data"]
    df = pl.scan_parquet("../data/cache/data.parquet").filter(
        period_tr | period_va
    )
    feature_cols = [c for c in df.collect_schema().names() if "feature" in c] + ["OpenInterest", "LogVolume", "LogAmount"]
    df = df.select(list(set([time_col, target_col] + feature_cols + categorical_cols + misc))).collect()
    
    my_scaler = CustomScaler(features=feature_cols, scale=True)
    _ = my_scaler.fit_transform(df.filter(period_tr & (~pl.col("DATA_is_bad_data"))))
    df = my_scaler.transform(df)

    lookback = 30
    model_params = {
        "num_numeric_features": len(feature_cols),
        "cat_cardinals": [cat_card_dict[k] for k in categorical_cols],
        "num_sectors": 7,
        
        "d_model_time": 256,
        "nhead_time": 8,
        "num_layers_time": 1,
        "dim_ff_time": 384,
        "dropout_time": 0.2,
        
        "use_rope_time": True,
        "time_is_causal": True,
        "time_pool": "last",
    
        "ffn_gate_type_time": "glu",        # "none"/"gelu", "glu", "geglu", "swiglu"
        "attn_gate_type_time": "element",      # "none", "head", "element"
    
        "num_gnn_layers": 0,
        "gnn_dim_ff_symbol": 256,
        "gnn_ffn_gate_symbol": "glu",
        "gnn_dim_ff_sector": 384,
        "gnn_ffn_gate_sector": "glu",
    
        "use_sector_attn": True,
        "sector_nhead": 8,
        "sector_attn_dropout": 0.2,
        "sector_attn_gate_type": "element",
        "gnn_agg_mode": "attn",
        
        "use_learned_cluster": True,
        "num_learned_clusters": 2,
        "cluster_topk": None,   # 每个 cluster 只用 top-k symbol 做聚合
        "sym_topk": None,       # 每个 symbol 只用 top-k cluster 做反向聚合
    }
    
    exp_name = f"cstr|lb={lookback}|"
    for k, v in model_params.items():
        if k in ["num_numeric_features", "cat_cardinals"]:
            continue
        exp_name = exp_name + f"{get_abbr(k)}={v}|"
        
    model = SymbolSectorBipartiteGNN(**model_params)
    
    T = lookback
    train_set = CrossSectionDataset(df=df.filter(period_tr & (~pl.col("DATA_is_bad_data"))), feature_cols=feature_cols, target_col=target_col, categorical_cols=categorical_cols, lookback=T, base_col=base_col, need_clip=True)
    val_set = CrossSectionDataset(df=df.filter(period_va), feature_cols=feature_cols, target_col=target_col, categorical_cols=categorical_cols, lookback=T, base_col=base_col)
    
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_set,
        batch_size=1024,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=cross_section_collate_fn
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1024,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=cross_section_collate_fn
    )
    loss_weights = {
        "mse": 1,
        "corr": 0,
        "custom": 0,
        "ret": 0,
        "custom_mse": 0,
        "sharpe": 0,
    }
    cfg = TrainConfig(
        epochs=200,
        ckpt_dir=os.path.join("../ddp_rolling/ckpt", exp_name),
        log_root="../ddp_rolling/exp_log",
        exp_name=exp_name,
        opt_type="AdamW",
        lr=1e-4,
        sched_type=None,
        use_amp=True,
        grad_accum=1,
        max_grad_norm=1e-3,
        weight_decay=1e-2,
        patience=2,
        loss_weights=loss_weights,
        log_param_hist=False,
        model_name="GNN",
        model_hparams=model_params,
    )

    trainer = DDPTrainer(model, cfg, rank=rank, world_size=world_size, local_rank=local_rank)
    trainer.fit(train_loader, val_loader)

    dist.destroy_process_group()

if __name__ == "__main__":
    # 例如：torchrun --nproc_per_node=4 train_xxx.py
    main()
