import os
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

from dl.dataio import CustomDataset, custom_collate_fn
from dl.models.M1 import MoETimeSeries
from dl.models.M2 import GRUBase, MultiScaleGRU
from dl.models.M3 import DualAxisTransformer
from dl.models.baseline_GRU import GRUBase
from dl.metrics import CorrLoss, RetLoss, CustomLoss
from dl.trainer import TrainConfig, Trainer
from dl.pipeline import RollingTrainingPipeline
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

if __name__ == "__main__":    
    set_global_seed()
    
    categorical_cols = ['Symbol', 'DATA_sector']
    time_col = "Time"
    target_col = "ret_30min"
    symbol_col = "Symbol"
    base_col = None
    misc = ["DATA_is_trading_hour", "DATA_is_last_30min"]
    df = pl.scan_parquet(f"data/cache/data.parquet")
    feature_cols = [c for c in df.collect_schema().names() if "feature" in c]
    df = df.select([time_col, target_col] + feature_cols + categorical_cols + misc)
    
    model = GRUBase(
        num_numeric_features=len(feature_cols),
        cat_cardinals=[52,7],
        hidden_size=256,
        num_layers=1,
        head_hidden=128,
    )
    lr = 1e-5
    cfg = TrainConfig(
        use_amp=True, 
        epochs=10,
        lr=lr,
        min_lr=5e-7,
        patience=0,
        sched_type=None,
        ckpt_dir=f"results/DualTransformer/formal_exp",
        max_grad_norm=1e4,
        weight_decay=1e-8/lr,
        T_max=20,
        loss_weights={
            "mse": 0,
            "corr": 0,
            "ret": 1e4,
            "custom": 0,
            "custom_mse": 1,
        }
    )
    trainer = Trainer(model, cfg)
    pipeline = RollingTrainingPipeline(
        model=model,
        trainer=trainer,
        lazy_df=df,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        preprocessor=CustomScaler(scale=True),
        clip_train=True,
        lookback=300
    )
    results = pipeline.run(datetime(2016,1,1), datetime(2021,3,1), timedelta(days=365))