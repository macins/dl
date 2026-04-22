from __future__ import annotations

import argparse
import json
import random
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import polars as pl

from jump_dl.src.config import load_config_with_inheritance
from jump_dl.src.dataio import build_slice_dataloader
from jump_dl.src.models import build_model
from jump_dl.src.objectives import CosineSimilarityObjective
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


def _build_dataloader(df: pl.DataFrame, dataset_cfg: Mapping[str, Any], loader_cfg: Mapping[str, Any], model_cfg: Mapping[str, Any]):
    feature_cols = _resolve_feature_cols(df, dataset_cfg, model_cfg)
    target_cols = _resolve_target_cols(dataset_cfg)
    cfg = dict(loader_cfg)
    return build_slice_dataloader(
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
    return parser.parse_args()


def _build_objective(
    cfg: Mapping[str, Any],
    *,
    dataset_cfg: Mapping[str, Any],
    target_stats: Mapping[str, Mapping[str, float]],
) -> CosineSimilarityObjective:
    objective_cfg = dict(cfg.get("objective", {}))
    target_key = str(objective_cfg.get("target_key", cfg.get("target_col", "ret_30min")))
    target_cols = _resolve_target_cols(dataset_cfg)
    inferred_target_index = objective_cfg.get("target_index")
    if inferred_target_index is None and target_key in target_cols:
        inferred_target_index = target_cols.index(target_key)
    stats = target_stats.get(target_key, {"mean": 0.0, "std": 1.0})
    return CosineSimilarityObjective(
        lam_cos=float(objective_cfg.get("lam_cos", 1.0)),
        lam_mse=float(objective_cfg.get("lam_mse", 1.0)),
        pred_key=str(objective_cfg.get("pred_key", cfg.get("target_col", "ret_30min"))),
        target_key=target_key,
        target_mean=float(stats.get("mean", 0.0)),
        target_std=float(stats.get("std", 1.0)),
        pred_index=objective_cfg.get("pred_index"),
        target_index=inferred_target_index,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config_with_inheritance(args.config)
    _set_seed(int(cfg.get("seed", 42)))

    data_cfg = dict(cfg["data"])
    train_dataset_cfg, val_dataset_cfg = _resolve_dataset_cfgs(data_cfg)
    model_cfg = _inject_vocab_sizes(dict(cfg["model"]), train_dataset_cfg)
    print("Building loaders...")
    train_df = _load_frame(train_dataset_cfg)
    val_df = _load_frame(val_dataset_cfg) if val_dataset_cfg is not None else None
    numeric_groups = [str(v) for v in model_cfg.get("numeric_feature_groups", ["continuous"])]
    feature_stats = _compute_feature_stats(
        train_df,
        _resolve_feature_cols(train_df, train_dataset_cfg, model_cfg),
        numeric_groups=numeric_groups,
    )
    target_stats = _compute_target_stats(train_df, _resolve_target_cols(train_dataset_cfg))
    train_loader = _build_dataloader(train_df, train_dataset_cfg, data_cfg.get("train_dataloader", {}), model_cfg)
    val_loader = (
        _build_dataloader(val_df, val_dataset_cfg, data_cfg.get("val_dataloader", {"shuffle": False}), model_cfg)
        if val_df is not None else None
    )
    print("Building model...")
    model = build_model(model_cfg)
    objective = _build_objective(cfg, dataset_cfg=train_dataset_cfg, target_stats=target_stats)
    optimizer = build_optimizer(cfg["optimizer"], model.parameters())
    scheduler = build_scheduler(cfg.get("scheduler"), optimizer)
    trainer = Trainer(
        model=model,
        objective=objective,
        optimizer=optimizer,
        scheduler=scheduler,
        feature_stats=feature_stats,
        config=TrainerConfig(**dict(cfg.get("trainer", {}))),
        output_dir=cfg.get("output_dir", "workdirs/jump_dl"),
    )
    print("Start training...")
    history = trainer.fit(train_loader, val_loader)
    output_dir = Path(cfg.get("output_dir", "workdirs/jump_dl"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] output_dir={output_dir} epochs={len(history)}", flush=True)


if __name__ == "__main__":
    main()
