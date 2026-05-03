from __future__ import annotations

import argparse
import json
import random
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl

import re
from datetime import date, datetime

from jump_dl.src.config import load_config_with_inheritance
from jump_dl.src.dataio import build_slice_dataloader, build_market_day_dataloader
from jump_dl.src.models import build_model
from jump_dl.src.objectives import CosineSimilarityObjective, MoGRegressionObjective
from jump_dl.src.optimizers import build_optimizer
from jump_dl.src.schedulers import build_scheduler
from jump_dl.src.trainer import Trainer, TrainerConfig
from jump_dl.src.utils.externals import ensure_torch
from jump_dl.src.utils.vocab import load_vocab, serialize_vocab_key

torch = ensure_torch()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _deep_merge_dict(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], Mapping) and isinstance(value, Mapping):
            out[key] = _deep_merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def _filter_frame(df: pl.DataFrame | pl.LazyFrame, config: Mapping[str, Any]) -> pl.DataFrame:
    time_col = str(config.get("time_col", "Time"))
    symbol_col = str(config.get("symbol_col", "Symbol"))
    start_date = config.get("start_date")
    end_date = config.get("end_date")
    years = config.get("years")
    symbols = config.get("symbols")

    expr = None
    if start_date is not None:
        cond = pl.col(time_col).dt.date() >= pl.lit(date.fromisoformat(str(start_date)))
        expr = cond if expr is None else expr & cond
    if end_date is not None:
        cond = pl.col(time_col).dt.date() <= pl.lit(date.fromisoformat(str(end_date)))
        expr = cond if expr is None else expr & cond
    if years:
        cond = pl.col(time_col).dt.year().is_in([int(v) for v in years])
        expr = cond if expr is None else expr & cond
    if symbols:
        cond = pl.col(symbol_col).is_in(list(symbols))
        expr = cond if expr is None else expr & cond
    return df if expr is None else df.filter(expr)


def _apply_vocab(df: pl.DataFrame, config: Mapping[str, Any]) -> pl.DataFrame:
    vocab_path = config.get("vocab_path")
    categorical_cols = [str(v) for v in config.get("categorical_cols", [])]
    if not vocab_path or not categorical_cols:
        return df

    vocab = load_vocab(vocab_path)
    exprs = []
    for col in categorical_cols:
        token_to_id = vocab.get(col, {})
        exprs.append(
            pl.col(col).map_elements(
                lambda value, token_to_id=token_to_id: int(token_to_id.get(serialize_vocab_key(value), 1)),
                return_dtype=pl.Int64,
            ).alias(col)
        )
    return df.with_columns(exprs)


def _resolve_target_cols(config: Mapping[str, Any]) -> list[str]:
    target_cols = config.get("target_cols")
    if target_cols is not None:
        return [str(v) for v in target_cols]
    return [str(config.get("target_col", "ret_30min"))]


def _load_frame(config: Mapping[str, Any]) -> pl.DataFrame:
    data_path = config.get("data_path")
    if data_path is None:
        raise ValueError("train_dataset.data_path is required for the new slice dataloader pipeline.")
    target_cols = _resolve_target_cols(config)
    expr = None
    for col in target_cols:
        cond = pl.col(col).is_not_null()
        expr = cond if expr is None else expr & cond
    df = pl.scan_parquet(data_path)
    
    if expr is not None:
        df = df.filter(expr)
    df = _filter_frame(df, config)
    df = _apply_vocab(df, config)
    return df.collect()


def _resolve_feature_cols(df: pl.DataFrame, dataset_cfg: Mapping[str, Any], model_cfg: Mapping[str, Any]) -> dict[str, list[str]]:
    explicit = dataset_cfg.get("feature_cols")
    if explicit is not None:
        return {str(k): [str(v) for v in values] for k, values in dict(explicit).items()}

    target_cols = set(_resolve_target_cols(dataset_cfg))
    categorical_cols = [str(v) for v in dataset_cfg.get("categorical_cols", [])]
    time_col = str(dataset_cfg.get("time_col", "Time"))
    symbol_col = str(dataset_cfg.get("symbol_col", "Symbol"))
    excluded = {time_col, symbol_col, *target_cols}

    numeric_feature_groups = [str(v) for v in model_cfg.get("numeric_feature_groups", ["continuous"])]
    if len(numeric_feature_groups) > 1:
        raise ValueError("Please provide data.train_dataset.feature_cols when using multiple numeric_feature_groups.")

    numeric_cols = [
        col for col, dtype in df.schema.items()
        if col not in excluded
        and col not in categorical_cols
        and hasattr(dtype, "is_numeric")
        and dtype.is_numeric()
    ]

    feature_cols: dict[str, list[str]] = {}
    if numeric_cols:
        feature_cols[numeric_feature_groups[0]] = numeric_cols
    if categorical_cols:
        feature_cols[str(model_cfg.get("categorical_group_name", "category"))] = categorical_cols
    if not feature_cols:
        raise ValueError("No feature columns were resolved. Please set data.train_dataset.feature_cols explicitly.")
    return feature_cols


def _compute_feature_stats(
    train_df: pl.DataFrame,
    feature_cols: Mapping[str, list[str]],
    *,
    numeric_groups: list[str],
) -> dict[str, dict[str, list[float]]]:
    stats: dict[str, dict[str, list[float]]] = {}
    for group_name, cols in feature_cols.items():
        if group_name not in numeric_groups:
            continue
        if not cols:
            continue
        frame = train_df.select(cols)
        means = frame.select([pl.col(col).mean().alias(col) for col in cols]).row(0, named=True)
        stds = frame.select([pl.col(col).std().alias(col) for col in cols]).row(0, named=True)
        stats[group_name] = {
            "mean": [float(means[col] if means[col] is not None else 0.0) for col in cols],
            "std": [float(stds[col] if stds[col] not in (None, 0.0) else 1.0) for col in cols],
        }
    return stats

def _unique_preserve_order(values: Sequence[str]) -> list[str]:
    seen = set()
    out = []
    for v in values:
        v = str(v)
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def _stats_by_col(
    feature_stats: Mapping[str, Mapping[str, list[float]]],
    *,
    group_name: str,
    cols: Sequence[str],
) -> dict[str, tuple[float, float]]:
    stats = feature_stats[group_name]
    means = stats["mean"]
    stds = stats["std"]

    if len(means) != len(cols) or len(stds) != len(cols):
        raise ValueError(
            f"Stats length mismatch for group={group_name!r}: "
            f"len(cols)={len(cols)}, len(mean)={len(means)}, len(std)={len(stds)}"
        )

    return {
        str(col): (float(mu), float(std if std not in (None, 0.0) else 1.0))
        for col, mu, std in zip(cols, means, stds)
    }


def _add_temp_normalized_cols(
    df: pl.DataFrame,
    *,
    source_cols: Sequence[str],
    stats_by_col: Mapping[str, tuple[float, float]],
    suffix: str = "__mkt_src_norm",
    eps: float = 1e-6,
    clip_value: float | None = 10.0,
) -> tuple[pl.DataFrame, list[str]]:
    exprs = []
    out_cols = []

    for col in source_cols:
        col = str(col)
        if col not in stats_by_col:
            raise ValueError(f"Missing stats for market_state source col: {col!r}")

        mu, std = stats_by_col[col]
        std = max(float(std), float(eps))

        out_col = f"{col}{suffix}"
        expr = (pl.col(col).cast(pl.Float64) - float(mu)) / std

        if clip_value is not None:
            expr = expr.clip(-float(clip_value), float(clip_value))

        exprs.append(
            expr
            .fill_nan(0.0)
            .fill_null(0.0)
            .cast(pl.Float32)
            .alias(out_col)
        )
        out_cols.append(out_col)

    return df.with_columns(exprs), out_cols


def _market_col_name(prefix: str, col: str, stat: str) -> str:
    # 避免原始列名太奇怪时污染 namespace
    return f"{prefix}__{col}__{stat}"


def _add_market_state_columns(
    df: pl.DataFrame,
    *,
    source_cols: Sequence[str],
    time_col: str = "Time",
    group_cols: Sequence[str] | None = None,
    prefix: str = "mkt",
    stats: Sequence[str] = ("mean", "std", "abs_mean", "rms", "pos_frac"),
    leave_one_out: bool = True,
    eps: float = 1e-6,
    fill_value: float = 0.0,
) -> tuple[pl.DataFrame, list[str]]:
    """
    source_cols 应该已经是 train-stat normalized 后的临时列。

    默认按同一个 Time 做全市场截面统计：
        group_cols = [time_col]

    输出列可以作为 batch["features"]["market_state"]。
    """
    source_cols = _unique_preserve_order(source_cols)
    stats = _unique_preserve_order(stats)

    if not source_cols:
        return df, []

    if group_cols is None:
        group_cols = [time_col]
    else:
        group_cols = [str(c) for c in group_cols]

    supported = {
        "mean",
        "std",
        "abs_mean",
        "rms",
        "skew",
        "kurt",
        "pos_frac",
        "zscore",
    }
    unknown = [s for s in stats if s not in supported]
    if unknown:
        raise ValueError(f"Unknown market_state stats={unknown}; supported={sorted(supported)}")

    n = pl.len().over(group_cols).cast(pl.Float64)

    def peer_mean(x: pl.Expr) -> pl.Expr:
        total = x.sum().over(group_cols)
        if leave_one_out:
            return (
                pl.when(n > 1.0)
                .then((total - x) / (n - 1.0))
                .otherwise(pl.lit(fill_value))
            )
        return (
            pl.when(n > 0.0)
            .then(total / n)
            .otherwise(pl.lit(fill_value))
        )

    def clean(expr: pl.Expr, name: str) -> pl.Expr:
        return (
            expr
            .fill_nan(fill_value)
            .fill_null(fill_value)
            .cast(pl.Float32)
            .alias(name)
        )

    exprs: list[pl.Expr] = []
    out_cols: list[str] = []

    for col in source_cols:
        col = str(col)
        x = pl.col(col).cast(pl.Float64)

        x2 = x * x
        x3 = x2 * x
        x4 = x2 * x2

        e1 = peer_mean(x)
        e2 = peer_mean(x2)
        e3 = peer_mean(x3)
        e4 = peer_mean(x4)

        var = e2 - e1 * e1
        var_pos = pl.when(var > eps).then(var).otherwise(pl.lit(eps))
        std = var_pos.sqrt()

        m3 = e3 - 3.0 * e1 * e2 + 2.0 * e1 * e1 * e1
        m4 = (
            e4
            - 4.0 * e1 * e3
            + 6.0 * e1 * e1 * e2
            - 3.0 * e1 * e1 * e1 * e1
        )

        stat_exprs: dict[str, pl.Expr] = {
            "mean": e1,
            "std": std,
            "abs_mean": peer_mean(x.abs()),
            "rms": peer_mean(x2).clip(lower_bound=eps).sqrt(),
            "skew": m3 / (std * var_pos),
            "kurt": m4 / (var_pos * var_pos) - 3.0,
            "pos_frac": peer_mean((x > 0.0).cast(pl.Float64)),
            "zscore": (x - e1) / std,
        }

        for stat in stats:
            out_col = _market_col_name(prefix, col, stat)
            exprs.append(clean(stat_exprs[stat], out_col))
            out_cols.append(out_col)

    return df.with_columns(exprs), out_cols


def _prepare_market_state_splits(
    *,
    train_df: pl.DataFrame,
    val_df: pl.DataFrame | None,
    feature_cols: dict[str, list[str]],
    feature_stats: dict[str, dict[str, list[float]]],
    dataset_cfg: Mapping[str, Any],
    data_cfg: Mapping[str, Any],
) -> tuple[pl.DataFrame, pl.DataFrame | None, dict[str, list[str]], dict[str, Any]]:
    """
    在 run_train 阶段给 train/val 加 market_state columns。

    返回：
        train_df, val_df, updated_feature_cols, market_info
    """
    market_cfg = dict(data_cfg.get("market_state", {}))
    if not bool(market_cfg.get("enabled", False)):
        return train_df, val_df, feature_cols, {"enabled": False}

    time_col = str(dataset_cfg.get("time_col", "Time"))

    source_group = str(market_cfg.get("source_group", "continuous"))
    group_name = str(market_cfg.get("group_name", "market_state"))

    if source_group not in feature_cols:
        raise ValueError(
            f"market_state.source_group={source_group!r} not in feature_cols. "
            f"Available groups: {list(feature_cols.keys())}"
        )

    if source_group not in feature_stats:
        raise ValueError(
            f"market_state source_group={source_group!r} has no feature_stats. "
            f"Make sure it is included in numeric_groups."
        )

    source_cols_cfg = market_cfg.get("source_cols")
    if source_cols_cfg is None:
        source_cols = list(feature_cols[source_group])
    else:
        source_cols = [str(c) for c in source_cols_cfg]

    source_cols = _unique_preserve_order(source_cols)

    missing = [c for c in source_cols if c not in feature_cols[source_group]]
    if missing:
        raise ValueError(
            f"market_state.source_cols contains columns not in source_group={source_group!r}: "
            f"{missing[:20]}"
        )

    stats_map = _stats_by_col(
        feature_stats,
        group_name=source_group,
        cols=feature_cols[source_group],
    )

    source_clip = market_cfg.get("source_clip_value", 10.0)
    eps = float(market_cfg.get("eps", 1e-6))
    temp_suffix = str(market_cfg.get("temp_suffix", "__mkt_src_norm"))

    train_df, train_norm_cols = _add_temp_normalized_cols(
        train_df,
        source_cols=source_cols,
        stats_by_col=stats_map,
        suffix=temp_suffix,
        eps=eps,
        clip_value=source_clip,
    )

    val_norm_cols = None
    if val_df is not None:
        val_df, val_norm_cols = _add_temp_normalized_cols(
            val_df,
            source_cols=source_cols,
            stats_by_col=stats_map,
            suffix=temp_suffix,
            eps=eps,
            clip_value=source_clip,
        )

    stats_list = [str(s) for s in market_cfg.get(
        "stats",
        ["mean", "std", "abs_mean", "rms", "pos_frac"],
    )]

    group_cols = market_cfg.get("group_cols")
    if group_cols is not None:
        group_cols = [str(c) for c in group_cols]

    prefix = str(market_cfg.get("prefix", "mkt"))
    leave_one_out = bool(market_cfg.get("leave_one_out", True))
    fill_value = float(market_cfg.get("fill_value", 0.0))

    train_df, market_cols = _add_market_state_columns(
        train_df,
        source_cols=train_norm_cols,
        time_col=time_col,
        group_cols=group_cols,
        prefix=prefix,
        stats=stats_list,
        leave_one_out=leave_one_out,
        eps=eps,
        fill_value=fill_value,
    )

    if val_df is not None:
        if val_norm_cols is None:
            raise RuntimeError("Internal error: val_norm_cols is None while val_df is not None.")
        val_df, val_market_cols = _add_market_state_columns(
            val_df,
            source_cols=val_norm_cols,
            time_col=time_col,
            group_cols=group_cols,
            prefix=prefix,
            stats=stats_list,
            leave_one_out=leave_one_out,
            eps=eps,
            fill_value=fill_value,
        )
        if val_market_cols != market_cols:
            raise RuntimeError("Train/val generated different market_state columns.")

    updated_feature_cols = dict(feature_cols)
    updated_feature_cols[group_name] = market_cols

    market_info = {
        "enabled": True,
        "group_name": group_name,
        "source_group": source_group,
        "source_cols": source_cols,
        "market_cols": market_cols,
        "num_market_cols": len(market_cols),
        "stats": stats_list,
        "leave_one_out": leave_one_out,
    }
    return train_df, val_df, updated_feature_cols, market_info

def _compute_target_stats(
    train_df: pl.DataFrame,
    target_cols: list[str],
) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for col in target_cols:
        agg = train_df.select(
            pl.col(col).mean().alias("mean"),
            pl.col(col).std().alias("std"),
        ).row(0, named=True)
        std = agg["std"]
        stats[col] = {
            "mean": float(agg["mean"]),
            "std": float(std),
        }
    return stats


def _build_dataloader(
    df: pl.DataFrame,
    dataset_cfg: Mapping[str, Any],
    loader_cfg: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
    *,
    feature_cols: Mapping[str, list[str]] | None = None,
):
    if feature_cols is None:
        feature_cols = _resolve_feature_cols(df, dataset_cfg, model_cfg)

    target_cols = _resolve_target_cols(dataset_cfg)
    cfg = dict(loader_cfg)

    sample_layout = str(dataset_cfg.get("sample_layout", "symbol_day"))

    common_kwargs = dict(
        df=df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        symbol_col=str(dataset_cfg.get("symbol_col", "Symbol")),
        time_col=str(dataset_cfg.get("time_col", "Time")),
        batch_size=int(cfg.pop("batch_size", 32)),
        shuffle=bool(cfg.pop("shuffle", True)),
        drop_last=bool(cfg.pop("drop_last", False)),
        num_workers=int(cfg.pop("num_workers", 0)),
        pin_memory=bool(cfg.pop("pin_memory", False)),
        prefetch_factor=int(cfg.pop("prefetch_factor", 2)),
        persistent_workers=bool(cfg.pop("persistent_workers", True)),
        sort=bool(dataset_cfg.get("sort", True)),
        share_memory=cfg.pop("share_memory", None),
        return_meta=bool(dataset_cfg.get("return_meta", True)),
    )

    if sample_layout in {"symbol_day", "slice"}:
        return build_slice_dataloader(**common_kwargs)

    if sample_layout in {"market_day", "day_market", "panel_day"}:
        return build_market_day_dataloader(
            **common_kwargs,
            validate_unique=bool(dataset_cfg.get("validate_unique", True)),
            vocab_path=dataset_cfg.get(
                "vocab_path",
                "/root/autodl-tmp/dl/jump_dl/artifacts/vocabs.pkl",
            ),
        )

    raise ValueError(
        f"Unknown dataset sample_layout={sample_layout!r}. "
        "Expected 'symbol_day' or 'market_day'."
    )


def _inject_vocab_sizes(model_cfg: dict[str, Any], dataset_cfg: Mapping[str, Any]) -> dict[str, Any]:
    vocab_path = dataset_cfg.get("vocab_path")
    if not vocab_path:
        return model_cfg
    vocab = load_vocab(vocab_path)
    model_cfg = dict(model_cfg)
    model_cfg["vocab_sizes"] = {col: max(ids.values(), default=1) + 1 for col, ids in vocab.items()}
    return model_cfg


def _resolve_dataset_cfgs(data_cfg: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if "dataset" in data_cfg:
        shared_dataset_cfg = dict(data_cfg["dataset"])
        split_cfg = data_cfg.get("split", {})
        if not isinstance(split_cfg, Mapping):
            raise TypeError("data.split must be a mapping when data.dataset is provided.")

        train_split_cfg = split_cfg.get("train", {})
        val_split_cfg = split_cfg.get("val")
        if train_split_cfg is not None and not isinstance(train_split_cfg, Mapping):
            raise TypeError("data.split.train must be a mapping.")
        if val_split_cfg is not None and not isinstance(val_split_cfg, Mapping):
            raise TypeError("data.split.val must be a mapping.")

        train_dataset_cfg = _deep_merge_dict(shared_dataset_cfg, dict(train_split_cfg or {}))
        val_dataset_cfg = (
            _deep_merge_dict(shared_dataset_cfg, dict(val_split_cfg))
            if val_split_cfg is not None else None
        )
        return train_dataset_cfg, val_dataset_cfg

    train_dataset_cfg = dict(data_cfg["train_dataset"])
    val_dataset_cfg = dict(data_cfg["val_dataset"]) if "val_dataset" in data_cfg else None
    return train_dataset_cfg, val_dataset_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run jump_dl training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", default=None, help="Optional human-readable run name.")
    parser.add_argument("--output-dir", default=None, help="Override base output directory.")
    parser.add_argument("--no-timestamp", action="store_true", help="Do not append timestamp to run directory.")
    return parser.parse_args()
    
def _sanitize_run_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9_.=-]+", "_", name)
    name = name.strip("._-")
    return name or "run"


def _resolve_run_output_dir(cfg: Mapping[str, Any], args: argparse.Namespace) -> Path:
    """
    Resolve one unique output directory for this training run.

    Supported config forms:

    output_dir: workdirs/jump_dl
    run_name: my_exp

    or:

    output_dir: workdirs/jump_dl
    run:
      name: my_exp
      timestamp: true
    """
    base_output_dir = Path(args.output_dir or cfg.get("output_dir", "workdirs/jump_dl"))

    run_cfg = cfg.get("run", {})
    if run_cfg is None:
        run_cfg = {}
    if not isinstance(run_cfg, Mapping):
        raise TypeError("run must be a mapping if provided.")

    run_name = args.run_name or run_cfg.get("name") or cfg.get("run_name")
    timestamp_enabled = (not args.no_timestamp) and bool(run_cfg.get("timestamp", True))

    parts: list[str] = []
    if run_name:
        parts.append(_sanitize_run_name(str(run_name)))
    if timestamp_enabled:
        parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Default behavior: always create a unique child directory.
    if not parts:
        parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))

    return base_output_dir / "__".join(parts)

def _build_objective(
    cfg: Mapping[str, Any],
    *,
    dataset_cfg: Mapping[str, Any],
    target_stats: Mapping[str, Mapping[str, float]],
) -> CosineSimilarityObjective:
    objective_cfg = dict(cfg.get("objective", {}))

    objective_name = str(
        objective_cfg.get("name", objective_cfg.get("type", "cosine_similarity"))
    ).lower()

    target_cols = _resolve_target_cols(dataset_cfg)

    default_target_key = (
        dataset_cfg.get("target_col")
        or cfg.get("target_col")
        or (target_cols[0] if target_cols else "ret_30min")
    )

    target_key = str(objective_cfg.get("target_key", default_target_key))
    pred_key = str(objective_cfg.get("pred_key", target_key))

    inferred_target_index = objective_cfg.get("target_index")
    if inferred_target_index is None and target_key in target_cols:
        inferred_target_index = target_cols.index(target_key)

    stats = target_stats.get(target_key, {"mean": 0.0, "std": 1.0})

    # Allow explicit config override.
    target_mean = float(objective_cfg.get("target_mean", stats.get("mean", 0.0)))
    target_std = float(objective_cfg.get("target_std", stats.get("std", 1.0)))
    if target_std == 0.0:
        target_std = 1.0

    common_kwargs = dict(
        lam_cos=float(objective_cfg.get("lam_cos", 1.0)),
        lam_mse=float(objective_cfg.get("lam_mse", 1.0)),
        pred_key=pred_key,
        target_key=target_key,
        target_mean=target_mean,
        target_std=target_std,
        pred_index=objective_cfg.get("pred_index"),
        target_index=inferred_target_index,

        # New MoG / weighted loss args.
        lam_mog_nll=float(objective_cfg.get("lam_mog_nll", 0.0)),
        lam_usage_kl=float(objective_cfg.get("lam_usage_kl", 0.0)),
        lam_scale_reg=float(objective_cfg.get("lam_scale_reg", 0.0)),
        aux_loss_weight=float(objective_cfg.get("aux_loss_weight", 1.0)),
        loss_weights=objective_cfg.get("loss_weights"),
        loss_schedules=objective_cfg.get("loss_schedules"),
        mog_key=str(objective_cfg.get("mog_key", "mog")),
        sigma_floor=float(objective_cfg.get("sigma_floor", 1e-4)),
        sigma_max=(
            None
            if objective_cfg.get("sigma_max") is None
            else float(objective_cfg.get("sigma_max"))
        ),
        use_sample_weight=bool(objective_cfg.get("use_sample_weight", False)),
        weight_key=str(objective_cfg.get("weight_key", "weight")),
        eps=float(objective_cfg.get("eps", 1e-12)),
        path_key=str(objective_cfg.get("path_key", "path")),
        pred_inc_key=str(objective_cfg.get("pred_inc_key", "pred_inc")),
        pred_cum_key=str(objective_cfg.get("pred_cum_key", "pred_cum")),
        target_inc_key=objective_cfg.get("target_inc_key"),
        target_inc_keys=objective_cfg.get("target_inc_keys"),
        target_inc_means=objective_cfg.get("target_inc_means"),
        target_inc_stds=objective_cfg.get("target_inc_stds"),
        target_cum_key=objective_cfg.get("target_cum_key"),
        aux_cum_huber_weight=float(objective_cfg.get("aux_cum_huber_weight", 0.0)),
        aux_inc_huber_weight=float(objective_cfg.get("aux_inc_huber_weight", 0.0)),
        aux_huber_delta=float(objective_cfg.get("aux_huber_delta", 1.0)),
        aux_horizon_weights=objective_cfg.get("aux_horizon_weights"),
        multi_horizon=objective_cfg.get("multi_horizon"),
    )

    inc_keys_cfg = objective_cfg.get("target_inc_keys")
    if inc_keys_cfg and common_kwargs["target_inc_means"] is None and common_kwargs["target_inc_stds"] is None:
        inc_keys = [str(v) for v in inc_keys_cfg]
        common_kwargs["target_inc_means"] = [
            float(target_stats.get(k, {}).get("mean", 0.0))
            for k in inc_keys
        ]
        common_kwargs["target_inc_stds"] = [
            float(target_stats.get(k, {}).get("std", 1.0) or 1.0)
            for k in inc_keys
        ]

    if objective_name in {
        "cosine_similarity",
        "cosine",
        "cos_mse",
        "weighted_cosine",
    }:
        objective = CosineSimilarityObjective(**common_kwargs)

    elif objective_name in {
        "mog",
        "mog_regression",
        "mixture_of_gaussian",
        "mixture_of_gaussians",
        "mog_nll",
    }:
        objective = MoGRegressionObjective(**common_kwargs)

    else:
        raise ValueError(
            f"Unknown objective name={objective_name!r}. "
            "Expected one of: cosine_similarity, mog_regression."
        )

    return objective


def main() -> None:
    args = parse_args()
    cfg = load_config_with_inheritance(args.config)
    run_output_dir = _resolve_run_output_dir(cfg, args)
    run_output_dir.mkdir(parents=True, exist_ok=False)
    cfg["output_dir"] = str(run_output_dir)
    
    _set_seed(int(cfg.get("seed", 42)))
    print(f"[run] output_dir={run_output_dir}", flush=True)

    data_cfg = dict(cfg["data"])
    train_dataset_cfg, val_dataset_cfg = _resolve_dataset_cfgs(data_cfg)
    model_cfg = _inject_vocab_sizes(dict(cfg["model"]), train_dataset_cfg)
    
    print("Building model...")
    model = build_model(model_cfg)
    print(model)
    train_df = _load_frame(train_dataset_cfg)
    target_stats = _compute_target_stats(train_df, _resolve_target_cols(train_dataset_cfg))
    objective = _build_objective(cfg, dataset_cfg=train_dataset_cfg, target_stats=target_stats)
    print(objective)
    optimizer = build_optimizer(cfg["optimizer"], model.parameters())
    scheduler = build_scheduler(cfg.get("scheduler"), optimizer)
    print("Building loaders...", flush=True)
    val_df = _load_frame(val_dataset_cfg) if val_dataset_cfg is not None else None
    
    numeric_groups = [str(v) for v in model_cfg.get("numeric_feature_groups", ["continuous"])]
    
    # 1. 先 resolve 原始 feature_cols，注意此时还没有 market_state columns
    feature_cols = _resolve_feature_cols(train_df, train_dataset_cfg, model_cfg)
    
    # 2. 先用 train_df 计算原始 continuous stats。
    #    这份 stats 用来：
    #    a) trainer normalize continuous
    #    b) market-state source normalization
    feature_stats = _compute_feature_stats(
        train_df,
        feature_cols,
        numeric_groups=numeric_groups,
    )
    trainer = Trainer(
        model=model,
        objective=objective,
        optimizer=optimizer,
        scheduler=scheduler,
        feature_stats=feature_stats,
        config=TrainerConfig(**dict(cfg.get("trainer", {}))),
        output_dir=run_output_dir,
    )
    # 3. 用 train-only continuous stats 生成 train/val 的 market_state columns
    train_df, val_df, feature_cols, market_info = _prepare_market_state_splits(
        train_df=train_df,
        val_df=val_df,
        feature_cols=feature_cols,
        feature_stats=feature_stats,
        dataset_cfg=train_dataset_cfg,
        data_cfg=data_cfg,
    )
    
    if market_info.get("enabled"):
        print(
            f"[market_state] group={market_info['group_name']} "
            f"sources={len(market_info['source_cols'])} "
            f"cols={market_info['num_market_cols']} "
            f"stats={market_info['stats']} "
            f"leave_one_out={market_info['leave_one_out']}",
            flush=True,
        )
    
        # 重要：
        # market_state 是 numeric feature group，需要让 trainer 也 normalize 它。
        # 但是不要一定把它加进 model_cfg['numeric_feature_groups']，
        # 否则你的 encoder 可能会把 market_state 也送进 EMA/diff bank。
        stat_numeric_groups = list(dict.fromkeys([*numeric_groups, market_info["group_name"]]))
    else:
        stat_numeric_groups = numeric_groups
    
    # 4. 重新计算 feature_stats，这次包括 market_state。
    #    continuous stats 会和前面一致；market_state stats 是新增的。
    feature_stats = _compute_feature_stats(
        train_df,
        feature_cols,
        numeric_groups=stat_numeric_groups,
    )
    
    train_loader = _build_dataloader(
        train_df,
        train_dataset_cfg,
        data_cfg.get("train_dataloader", {}),
        model_cfg,
        feature_cols=feature_cols,
    )
    
    val_loader = (
        _build_dataloader(
            val_df,
            val_dataset_cfg,
            data_cfg.get("val_dataloader", {"shuffle": False}),
            model_cfg,
            feature_cols=feature_cols,
        )
        if val_df is not None else None
    )

    print("Start training...")
    cfg["resolved_feature_cols"] = feature_cols
    cfg["resolved_feature_stats_groups"] = list(feature_stats.keys())
    cfg["market_state_info"] = market_info
    history = trainer.fit(train_loader, val_loader)
    (run_output_dir / "resolved_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] output_dir={run_output_dir} epochs={len(history)}", flush=True)


if __name__ == "__main__":
    main()
