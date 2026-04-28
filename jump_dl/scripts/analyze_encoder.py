from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jump_dl.src.config import load_config_with_inheritance
from jump_dl.src.dataio import build_slice_dataloader
from jump_dl.src.models import build_model
from jump_dl.src.utils.externals import ensure_torch
from jump_dl.src.utils.vocab import load_vocab, serialize_vocab_key

torch = ensure_torch()


# ============================================================
# Config / data utilities
# ============================================================

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


def _filter_frame(df: pl.DataFrame | pl.LazyFrame, config: Mapping[str, Any]) -> pl.DataFrame | pl.LazyFrame:
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


def _resolve_target_cols(config: Mapping[str, Any]) -> list[str]:
    target_cols = config.get("target_cols")
    if target_cols is not None:
        return [str(v) for v in target_cols]
    return [str(config.get("target_col", "ret_30min"))]


def _apply_vocab(df: pl.DataFrame, config: Mapping[str, Any]) -> pl.DataFrame:
    """
    Compatibility with older training script.

    Important:
    Your current uploaded dataio already applies a hard-coded vocabs.pkl inside
    SliceBatchDataset. If that is the path used in training, you probably want
    to keep --apply-config-vocab OFF to avoid double encoding.
    """
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


def _load_frame(config: Mapping[str, Any], *, apply_config_vocab: bool) -> pl.DataFrame:
    data_path = config.get("data_path")
    if data_path is None:
        raise ValueError("dataset.data_path is required.")

    target_cols = _resolve_target_cols(config)

    expr = None
    for col in target_cols:
        cond = pl.col(col).is_not_null()
        expr = cond if expr is None else expr & cond

    df = pl.scan_parquet(data_path)

    if expr is not None:
        df = df.filter(expr)

    df = _filter_frame(df, config)
    out = df.collect()

    if apply_config_vocab:
        out = _apply_vocab(out, config)

    return out


def _resolve_feature_cols(
    df: pl.DataFrame,
    dataset_cfg: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
) -> dict[str, list[str]]:
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
        raise ValueError(
            "Please provide explicit feature_cols when using multiple numeric_feature_groups."
        )

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
        feature_cols[str(model_cfg.get("categorical_group_name", "categorical"))] = categorical_cols

    if not feature_cols:
        raise ValueError("No feature columns were resolved. Please set feature_cols explicitly.")

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

        mean = agg["mean"]
        std = agg["std"]

        stats[col] = {
            "mean": float(mean if mean is not None else 0.0),
            "std": float(std if std not in (None, 0.0) else 1.0),
        }

    return stats


def _build_dataloader(
    df: pl.DataFrame,
    dataset_cfg: Mapping[str, Any],
    loader_cfg: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
    *,
    batch_size: int | None,
    num_workers: int | None,
) -> Any:
    feature_cols = _resolve_feature_cols(df, dataset_cfg, model_cfg)
    target_cols = _resolve_target_cols(dataset_cfg)

    cfg = dict(loader_cfg)

    if batch_size is not None:
        cfg["batch_size"] = batch_size

    if num_workers is not None:
        cfg["num_workers"] = num_workers

    cfg["shuffle"] = False
    cfg["drop_last"] = False

    if int(cfg.get("num_workers", 0)) == 0:
        cfg["persistent_workers"] = False

    return build_slice_dataloader(
        df=df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        symbol_col=str(dataset_cfg.get("symbol_col", "Symbol")),
        time_col=str(dataset_cfg.get("time_col", "Time")),
        batch_size=int(cfg.pop("batch_size", 32)),
        shuffle=bool(cfg.pop("shuffle", False)),
        drop_last=bool(cfg.pop("drop_last", False)),
        num_workers=int(cfg.pop("num_workers", 0)),
        pin_memory=bool(cfg.pop("pin_memory", False)),
        prefetch_factor=int(cfg.pop("prefetch_factor", 2)),
        persistent_workers=bool(cfg.pop("persistent_workers", False)),
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
    model_cfg["vocab_sizes"] = {
        col: max(ids.values(), default=1) + 1
        for col, ids in vocab.items()
    }
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
            if val_split_cfg is not None
            else None
        )

        return train_dataset_cfg, val_dataset_cfg

    train_dataset_cfg = dict(data_cfg["train_dataset"])
    val_dataset_cfg = dict(data_cfg["val_dataset"]) if "val_dataset" in data_cfg else None
    return train_dataset_cfg, val_dataset_cfg


# ============================================================
# Tensor / metric helpers
# ============================================================

def _to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)

    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_to_device(v, device) for v in obj]

    if isinstance(obj, tuple):
        return tuple(_to_device(v, device) for v in obj)

    return obj


def _detach_to_cpu_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def _flatten_last_dim(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.shape[-1])


def _normalize_batch_features(
    batch: dict[str, Any],
    feature_stats: Mapping[str, Mapping[str, list[float]]],
    numeric_groups: list[str],
    device: torch.device,
) -> dict[str, Any]:
    for group in numeric_groups:
        x = batch["features"].get(group)

        if x is None:
            continue

        stats = feature_stats.get(group)

        if not stats:
            continue

        mean = torch.tensor(stats["mean"], device=device, dtype=x.dtype)
        std = torch.tensor(stats["std"], device=device, dtype=x.dtype).clamp_min(1e-12)

        batch["features"][group] = (x - mean) / std

    return batch


def _extract_pred(out: Any, pred_key: str | None, target_key: str) -> torch.Tensor:
    return out["preds"]["ret_30min"]
    if torch.is_tensor(out):
        return out

    if not isinstance(out, Mapping):
        raise TypeError(f"Model output must be tensor or mapping, got {type(out)}")

    candidate_keys = []

    if pred_key is not None:
        candidate_keys.append("preds")

    candidate_keys.extend([
        target_key,
        "pred",
        "prediction",
        "y_pred",
        "output",
        "preds",
        "logits",
    ])

    for key in candidate_keys:
        print(key)
        if key in out and torch.is_tensor(out[key]):
            return out[key]

    for value in out.values():
        if torch.is_tensor(value):
            return value

    raise KeyError(f"Could not extract prediction from model output keys={list(out.keys())}")


def _extract_target(
    batch: dict[str, Any],
    target_key: str,
    target_index: int | None,
) -> torch.Tensor | None:
    targets = batch.get("targets")

    if targets is None:
        return None

    if torch.is_tensor(targets):
        y = targets.float()

        if y.ndim >= 1 and y.shape[-1] > 1:
            if target_index is None:
                target_index = 0
            y = y[..., target_index:target_index + 1]

        return y

    if isinstance(targets, Mapping):
        if target_key in targets and torch.is_tensor(targets[target_key]):
            y = targets[target_key].float()
            if y.ndim >= 1 and y.shape[-1] > 1 and target_index is not None:
                y = y[..., target_index:target_index + 1]
            return y

        for value in targets.values():
            if torch.is_tensor(value):
                y = value.float()
                if y.ndim >= 1 and y.shape[-1] > 1 and target_index is not None:
                    y = y[..., target_index:target_index + 1]
                return y

    return None


def _extract_weight(batch: dict[str, Any], like: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(like, dtype=torch.float32)


def _ensure_pred_target_weight_shapes(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = pred.float()
    target = target.float()
    weight = weight.float()

    if pred.ndim == target.ndim - 1 and target.shape[-1] == 1:
        pred = pred.unsqueeze(-1)

    if target.ndim == pred.ndim - 1 and pred.shape[-1] == 1:
        target = target.unsqueeze(-1)

    if weight.ndim == pred.ndim - 1 and pred.shape[-1] == 1:
        weight = weight.unsqueeze(-1)

    if target.shape != pred.shape:
        try:
            target = torch.broadcast_to(target, pred.shape)
        except RuntimeError:
            target = target.reshape_as(pred)

    if weight.shape != pred.shape:
        try:
            weight = torch.broadcast_to(weight, pred.shape)
        except RuntimeError:
            weight = weight.reshape_as(pred)

    return pred, target, weight


def _valid_flat_mask_from_padding(batch: dict[str, Any], x: torch.Tensor) -> torch.Tensor | None:
    valid_mask = batch.get("padding_mask")

    if valid_mask is None or not torch.is_tensor(valid_mask):
        return None

    valid = valid_mask.bool()

    if x.ndim >= 3:
        expected = int(np.prod(x.shape[:-1]))
    elif x.ndim == 2:
        expected = int(np.prod(x.shape))
    else:
        return None

    if valid.numel() == expected:
        return valid.reshape(-1)

    return None


def _masked_flatten_pred_target_weight(
    batch: dict[str, Any],
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    ignore_padding_mask: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred, target, weight = _ensure_pred_target_weight_shapes(pred, target, weight)

    valid = None if ignore_padding_mask else _valid_flat_mask_from_padding(batch, pred)

    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    weight_flat = weight.reshape(-1)

    if valid is not None:
        if pred.ndim >= 3 and pred.shape[-1] == 1:
            pred_flat = pred_flat[valid]
            target_flat = target_flat[valid]
            weight_flat = weight_flat[valid]
        elif pred.ndim == 2:
            pred_flat = pred_flat[valid]
            target_flat = target_flat[valid]
            weight_flat = weight_flat[valid]

    return pred_flat, target_flat, weight_flat


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if a.size < 2:
        return float("nan")

    sa = a.std()
    sb = b.std()

    if sa == 0 or sb == 0:
        return float("nan")

    return float(np.corrcoef(a, b)[0, 1])


def _weighted_metrics(pred: np.ndarray, target: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    pred = pred.reshape(-1).astype(np.float64)
    target = target.reshape(-1).astype(np.float64)
    weight = weight.reshape(-1).astype(np.float64)

    good = np.isfinite(pred) & np.isfinite(target) & np.isfinite(weight) & (weight >= 0)

    pred = pred[good]
    target = target[good]
    weight = weight[good]

    if pred.size == 0:
        return {
            "mse": float("nan"),
            "weighted_r2_zero_benchmark": float("nan"),
            "weighted_cosine": float("nan"),
            "corr": float("nan"),
            "n": 0.0,
        }

    wsum = max(float(weight.sum()), 1e-12)

    mse = float((weight * (pred - target) ** 2).sum() / wsum)

    target_power = float((weight * target ** 2).sum() / wsum)
    r2 = float(1.0 - mse / max(target_power, 1e-12))

    cosine_denom = math.sqrt(
        max(float((weight * pred ** 2).sum()), 1e-12)
        * max(float((weight * target ** 2).sum()), 1e-12)
    )
    cosine = float((weight * pred * target).sum() / cosine_denom)

    corr = _safe_corr(pred, target)

    return {
        "mse": mse,
        "weighted_r2_zero_benchmark": r2,
        "weighted_cosine": cosine,
        "corr": corr,
        "n": float(pred.size),
    }


def _infer_target_key_and_index(
    cfg: Mapping[str, Any],
    dataset_cfg: Mapping[str, Any],
) -> tuple[str, int, str | None]:
    objective_cfg = dict(cfg.get("objective", {}))

    target_key = str(objective_cfg.get("target_key", cfg.get("target_col", "ret_30min")))

    pred_key = objective_cfg.get("pred_key", cfg.get("target_col", target_key))
    pred_key = str(pred_key) if pred_key is not None else None

    target_cols = _resolve_target_cols(dataset_cfg)

    target_index = objective_cfg.get("target_index")

    if target_index is None and target_key in target_cols:
        target_index = target_cols.index(target_key)

    if target_index is None:
        target_index = 0

    return target_key, int(target_index), pred_key


# ============================================================
# Checkpoint / model helpers
# ============================================================

def _load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    prefer_ema: bool = True,
) -> dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(ckpt, Mapping):
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

    candidate_keys = []

    if prefer_ema:
        candidate_keys.extend([
            "ema_model_state_dict",
            "ema_state_dict",
            "model_ema",
            "ema_model",
        ])

    candidate_keys.extend([
        "model_state_dict",
        "state_dict",
        "model",
        "module",
    ])

    state = None
    used_key = None

    for key in candidate_keys:
        if key in ckpt and isinstance(ckpt[key], Mapping):
            state = ckpt[key]
            used_key = key
            break

    if state is None:
        state = ckpt
        used_key = "<root>"

    clean_state = {}

    for key, value in state.items():
        if not torch.is_tensor(value):
            continue

        new_key = str(key)

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        clean_state[new_key] = value

    missing, unexpected = model.load_state_dict(clean_state, strict=False)

    return {
        "checkpoint_path": str(checkpoint_path),
        "used_key": used_key,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def _get_input_projection(model: torch.nn.Module) -> torch.nn.Module:
    if not hasattr(model, "encoder"):
        raise AttributeError("Model has no .encoder attribute.")

    encoder = model.encoder

    if not hasattr(encoder, "input_projection"):
        raise AttributeError("model.encoder has no .input_projection attribute.")

    return encoder.input_projection


def _has_ema_bank(model: torch.nn.Module) -> bool:
    return (
        hasattr(model, "encoder")
        and hasattr(model.encoder, "causal_ema_bank")
        and hasattr(model.encoder.causal_ema_bank, "convs")
    )


def _get_ema_bank(model: torch.nn.Module) -> torch.nn.Module | None:
    if not _has_ema_bank(model):
        return None
    return model.encoder.causal_ema_bank


# ============================================================
# IO / plotting helpers
# ============================================================

def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_barh(
    path: Path,
    labels: list[str],
    values: np.ndarray,
    title: str,
    xlabel: str,
    *,
    top_k: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    values = np.asarray(values)
    order = np.argsort(values)[::-1][:top_k]

    show_labels = [labels[i] for i in order][::-1]
    show_values = values[order][::-1]

    plt.figure(figsize=(10, max(4, 0.28 * len(show_labels))))
    plt.barh(np.arange(len(show_labels)), show_values)
    plt.yticks(np.arange(len(show_labels)), show_labels, fontsize=7)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _save_line(path: Path, values: np.ndarray, title: str, xlabel: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(np.asarray(values))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 2:
        raise ValueError(f"PCA expects 2D array, got shape={x.shape}")

    x = x - x.mean(axis=0, keepdims=True)

    u, s, _vt = np.linalg.svd(x, full_matrices=False)

    coords = u[:, :2] * s[:2]

    denom = max(float((s ** 2).sum()), 1e-12)
    explained = (s[:2] ** 2) / denom

    return coords, explained


def _save_scatter(
    path: Path,
    coords: np.ndarray,
    title: str,
    xlabel: str = "PC1",
    ylabel: str = "PC2",
    color: np.ndarray | None = None,
    annotate: list[str] | None = None,
    annotate_indices: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))

    if color is None:
        plt.scatter(coords[:, 0], coords[:, 1], s=10, alpha=0.7)
    else:
        plt.scatter(coords[:, 0], coords[:, 1], s=10, alpha=0.7, c=color)
        plt.colorbar()

    if annotate is not None and annotate_indices is not None:
        for i in annotate_indices:
            plt.text(coords[i, 0], coords[i, 1], annotate[i], fontsize=6)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _save_heatmap(
    path: Path,
    mat: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    xticklabels: list[str] | None = None,
    yticklabels: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    mat = np.asarray(mat)

    plt.figure(figsize=(max(8, 0.25 * mat.shape[1]), max(5, 0.20 * mat.shape[0])))
    plt.imshow(mat, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if xticklabels is not None and len(xticklabels) == mat.shape[1]:
        plt.xticks(np.arange(mat.shape[1]), xticklabels, rotation=90, fontsize=6)

    if yticklabels is not None and len(yticklabels) == mat.shape[0]:
        plt.yticks(np.arange(mat.shape[0]), yticklabels, fontsize=6)

    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _kmeans(x: np.ndarray, k: int, *, seed: int, n_iter: int = 50) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)

    n = x.shape[0]

    if n == 0:
        return np.zeros((0,), dtype=np.int64)

    k = max(1, min(int(k), n))

    rng = np.random.default_rng(seed)
    centers = x[rng.choice(n, size=k, replace=False)].copy()

    labels = np.zeros(n, dtype=np.int64)

    for _ in range(n_iter):
        dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        new_labels = dist.argmin(axis=1)

        if np.array_equal(new_labels, labels):
            break

        labels = new_labels

        for c in range(k):
            idx = labels == c
            if idx.any():
                centers[c] = x[idx].mean(axis=0)

    return labels


def _cosine_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


# ============================================================
# CNN / EMA bank helpers
# ============================================================

def _ema_init_kernel(span: int, *, dtype=np.float64) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    lags = np.arange(span - 1, -1, -1, dtype=dtype)
    kernel = alpha * (1.0 - alpha) ** lags
    kernel = kernel / max(float(kernel.sum()), 1e-12)
    return kernel.astype(dtype)


def _kernel_effective_stats(kernel: np.ndarray) -> dict[str, float]:
    k = np.asarray(kernel, dtype=np.float64)
    span = len(k)

    lags = np.arange(span - 1, -1, -1, dtype=np.float64)

    abs_sum = max(float(np.abs(k).sum()), 1e-12)
    signed_sum = float(k.sum())

    abs_center_lag = float((np.abs(k) * lags).sum() / abs_sum)

    pos_mass = float(np.clip(k, 0, None).sum())
    neg_mass = float(np.clip(-k, 0, None).sum())

    normalized_abs = np.abs(k) / abs_sum
    entropy = float(-(normalized_abs * np.log(normalized_abs + 1e-12)).sum())

    nonnegative_frac = float((k >= 0).mean())
    monotone_to_current_frac = float((np.diff(k) >= -1e-8).mean()) if span > 1 else 1.0

    current_weight = float(k[-1])
    oldest_weight = float(k[0])

    dc_gain = signed_sum
    highpass_score = float(1.0 - min(abs(dc_gain) / abs_sum, 1.0))

    return {
        "signed_sum": signed_sum,
        "abs_sum": abs_sum,
        "pos_mass": pos_mass,
        "neg_mass": neg_mass,
        "neg_mass_ratio": neg_mass / abs_sum,
        "abs_center_lag": abs_center_lag,
        "entropy": entropy,
        "nonnegative_frac": nonnegative_frac,
        "monotone_to_current_frac": monotone_to_current_frac,
        "current_weight": current_weight,
        "oldest_weight": oldest_weight,
        "dc_gain": dc_gain,
        "highpass_score": highpass_score,
        "l2_norm": float(np.linalg.norm(k)),
        "max_abs": float(np.abs(k).max()),
    }


def _kernel_type_label(stats: Mapping[str, float]) -> str:
    neg = float(stats["neg_mass_ratio"])
    mono = float(stats["monotone_to_current_frac"])
    nonneg = float(stats["nonnegative_frac"])
    highpass = float(stats["highpass_score"])

    if neg < 0.02 and nonneg > 0.98 and mono > 0.98:
        return "ema_like_lowpass"
    if highpass > 0.8 and neg > 0.10:
        return "difference_highpass"
    if neg > 0.25:
        return "signed_mixed_filter"
    if mono < 0.6:
        return "nonmonotone_lowpass_or_bandpass"
    return "weakly_deformed_ema"


def _save_kernel_plot(
    path: Path,
    kernel: np.ndarray,
    init_kernel: np.ndarray | None,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(kernel))

    plt.figure(figsize=(7, 4))
    plt.plot(x, kernel, marker="o", label="learned")

    if init_kernel is not None:
        plt.plot(x, init_kernel, marker="x", linestyle="--", label="ema_init")

    plt.xlabel("kernel index: oldest -> current")
    plt.ylabel("weight")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


# ============================================================
# Projection analysis
# ============================================================

def analyze_projection(
    *,
    model: torch.nn.Module,
    feature_names: list[str],
    output_dir: Path,
    top_k: int,
    n_clusters: int,
    seed: int,
) -> dict[str, Any]:
    figs = output_dir / "figures"
    tables = output_dir / "tables"
    arrays = output_dir / "arrays"

    for p in [figs, tables, arrays]:
        p.mkdir(parents=True, exist_ok=True)

    projection = _get_input_projection(model)

    W = projection.weight.detach().float().cpu().numpy()

    bias = None
    if getattr(projection, "bias", None) is not None:
        bias = projection.bias.detach().float().cpu().numpy()

    np.save(arrays / "input_projection_weight.npy", W)

    if bias is not None:
        np.save(arrays / "input_projection_bias.npy", bias)

    hidden_dim, projection_input_dim = W.shape

    # If CNN bank exists, W input dim is not equal to num raw features.
    # For raw feature loading, only analyze the first raw block if include_raw_numeric=True.
    if _has_ema_bank(model):
        encoder = model.encoder
        num_features = int(getattr(encoder, "num_features", len(feature_names)))
        include_raw_numeric = bool(getattr(encoder, "include_raw_numeric", True))

        if include_raw_numeric:
            raw_W = W[:, :num_features]
        else:
            raw_W = np.zeros((hidden_dim, num_features), dtype=W.dtype)

        if len(feature_names) != num_features:
            raise ValueError(
                f"feature_names length mismatch: len(feature_names)={len(feature_names)}, "
                f"encoder.num_features={num_features}."
            )
    else:
        raw_W = W
        if len(feature_names) != projection_input_dim:
            raise ValueError(
                f"feature_names length mismatch: len(feature_names)={len(feature_names)}, "
                f"but input_projection.weight.shape={W.shape}. "
                "Check feature_cols / numeric_feature_groups."
            )

    V = raw_W.T

    feature_l2 = np.linalg.norm(V, axis=1)
    feature_l1 = np.abs(V).sum(axis=1)
    feature_max_abs = np.abs(V).max(axis=1)

    rows = []

    for j, name in enumerate(feature_names):
        top_dim = int(np.abs(V[j]).argmax()) if V.shape[1] > 0 else 0

        rows.append({
            "feature": name,
            "feature_index": j,
            "raw_branch_weight_l2": float(feature_l2[j]),
            "raw_branch_weight_l1": float(feature_l1[j]),
            "raw_branch_weight_max_abs": float(feature_max_abs[j]),
            "top_hidden_dim_abs": top_dim,
            "top_hidden_loading": float(V[j, top_dim]) if V.shape[1] > 0 else 0.0,
        })

    rows = sorted(rows, key=lambda r: r["raw_branch_weight_l2"], reverse=True)

    _write_csv(tables / "feature_raw_branch_weight_importance.csv", rows)

    _save_barh(
        figs / "top_feature_raw_branch_weight_l2.png",
        feature_names,
        feature_l2,
        title="Top raw features by ||W_raw[:, j]||2",
        xlabel="L2 norm of raw feature loading",
        top_k=min(top_k, len(feature_names)),
    )

    hidden_l2 = np.linalg.norm(W, axis=1)
    hidden_l1 = np.abs(W).sum(axis=1)
    hidden_max_abs = np.abs(W).max(axis=1)

    hidden_rows = []

    for d in range(hidden_dim):
        row = W[d]
        top_pos = np.argsort(row)[::-1][:10]
        top_neg = np.argsort(row)[:10]

        hidden_rows.append({
            "hidden_dim": d,
            "weight_l2": float(hidden_l2[d]),
            "weight_l1": float(hidden_l1[d]),
            "weight_max_abs": float(hidden_max_abs[d]),
            "bias": float(bias[d]) if bias is not None else 0.0,
            "top_pos_projection_cols": "|".join(str(i) for i in top_pos),
            "top_pos_loadings": "|".join(f"{row[i]:.6g}" for i in top_pos),
            "top_neg_projection_cols": "|".join(str(i) for i in top_neg),
            "top_neg_loadings": "|".join(f"{row[i]:.6g}" for i in top_neg),
        })

    _write_csv(tables / "hidden_channel_summary.csv", hidden_rows)

    _save_line(
        figs / "hidden_channel_weight_l2.png",
        hidden_l2,
        title="Hidden channel loading norm",
        xlabel="hidden channel",
        ylabel="||W[d, :]||2",
    )

    if V.shape[0] >= 2 and V.shape[1] >= 2:
        V_norm = _cosine_normalize_rows(V)
        feature_coords, feature_explained = _pca_2d(V_norm)
        feature_cluster = _kmeans(V_norm, n_clusters, seed=seed)

        top_annotate = np.argsort(feature_l2)[::-1][:min(top_k, len(feature_names))]

        _save_scatter(
            figs / "feature_raw_loading_pca.png",
            feature_coords,
            title=f"Raw feature loading PCA; explained={feature_explained[0]:.2%},{feature_explained[1]:.2%}",
            annotate=feature_names,
            annotate_indices=top_annotate,
        )

        cluster_rows = []

        for j, name in enumerate(feature_names):
            cluster_rows.append({
                "feature": name,
                "feature_index": j,
                "cluster": int(feature_cluster[j]),
                "pc1": float(feature_coords[j, 0]),
                "pc2": float(feature_coords[j, 1]),
                "raw_branch_weight_l2": float(feature_l2[j]),
            })

        cluster_rows = sorted(cluster_rows, key=lambda r: (r["cluster"], -r["raw_branch_weight_l2"]))
        _write_csv(tables / "feature_raw_loading_clusters.csv", cluster_rows)

        feature_pca_explained = feature_explained.tolist()
    else:
        feature_pca_explained = None

    W_norm = _cosine_normalize_rows(W)
    hidden_coords, hidden_explained = _pca_2d(W_norm)

    _save_scatter(
        figs / "hidden_channel_pca.png",
        hidden_coords,
        title=f"Hidden channel PCA; explained={hidden_explained[0]:.2%},{hidden_explained[1]:.2%}",
        annotate=[str(i) for i in range(hidden_dim)],
        annotate_indices=np.argsort(hidden_l2)[::-1][:min(top_k, hidden_dim)],
    )

    return {
        "weight_shape": list(W.shape),
        "has_bias": bias is not None,
        "top_features_by_raw_branch_weight_l2": rows[:min(20, len(rows))],
        "top_hidden_channels_by_weight_l2": sorted(hidden_rows, key=lambda r: r["weight_l2"], reverse=True)[:20],
        "feature_raw_pca_explained": feature_pca_explained,
        "hidden_pca_explained": hidden_explained.tolist(),
    }


# ============================================================
# CNN / EMA bank analysis
# ============================================================

def analyze_causal_ema_bank(
    *,
    model: torch.nn.Module,
    feature_names: list[str],
    output_dir: Path,
    top_k: int,
) -> dict[str, Any]:
    figs = output_dir / "figures"
    tables = output_dir / "tables"
    arrays = output_dir / "arrays"
    kernel_figs = figs / "cnn_kernel_examples"

    for p in [figs, tables, arrays, kernel_figs]:
        p.mkdir(parents=True, exist_ok=True)

    bank = _get_ema_bank(model)

    if bank is None:
        return {
            "status": "skipped_no_causal_ema_bank",
        }

    spans = [int(s) for s in getattr(bank, "spans", [])]
    num_features = int(getattr(bank, "num_features", len(feature_names)))
    include_raw_numeric = bool(getattr(model.encoder, "include_raw_numeric", True))

    if num_features != len(feature_names):
        raise ValueError(
            f"EMA bank num_features={num_features}, but len(feature_names)={len(feature_names)}. "
            "Check encoder.num_features and feature column resolution."
        )

    projection = _get_input_projection(model)
    W = projection.weight.detach().float().cpu().numpy()

    hidden_dim, projection_input_dim = W.shape

    n_spans = len(spans)
    raw_dim = num_features if include_raw_numeric else 0
    ema_dim = num_features * n_spans

    if projection_input_dim < raw_dim + ema_dim:
        raise ValueError(
            f"input_projection input dim={projection_input_dim}, but expected at least "
            f"raw_dim + ema_dim = {raw_dim + ema_dim}. "
            "Maybe LazyLinear was initialized with a different encoder layout."
        )

    # ------------------------------------------------------------
    # 1. Kernel analysis
    # ------------------------------------------------------------

    kernel_rows: list[dict[str, Any]] = []

    drift_by_span = []
    cosine_by_span = []
    center_lag_by_span = []
    neg_ratio_by_span = []

    for span_idx, (span, conv) in enumerate(zip(spans, bank.convs)):
        weight = conv.weight.detach().float().cpu().numpy()
        kernels = weight[:, 0, :]

        np.save(arrays / f"cnn_kernel_span_{span}.npy", kernels)

        init = _ema_init_kernel(span)
        init_norm = max(float(np.linalg.norm(init)), 1e-12)

        span_drift = []
        span_cos = []
        span_center = []
        span_neg = []

        for j, feature in enumerate(feature_names):
            k = kernels[j].astype(np.float64)

            stats = _kernel_effective_stats(k)

            l2_drift = float(np.linalg.norm(k - init))
            cos_to_init = float(np.dot(k, init) / max(float(np.linalg.norm(k)) * init_norm, 1e-12))

            label = _kernel_type_label(stats)

            kernel_rows.append({
                "feature": feature,
                "feature_index": j,
                "span": span,
                "span_index": span_idx,
                "kernel_type": label,
                "cos_to_ema_init": cos_to_init,
                "l2_drift_from_ema_init": l2_drift,
                **stats,
            })

            span_drift.append(l2_drift)
            span_cos.append(cos_to_init)
            span_center.append(stats["abs_center_lag"])
            span_neg.append(stats["neg_mass_ratio"])

        drift_by_span.append(float(np.mean(span_drift)))
        cosine_by_span.append(float(np.mean(span_cos)))
        center_lag_by_span.append(float(np.mean(span_center)))
        neg_ratio_by_span.append(float(np.mean(span_neg)))

    kernel_rows_sorted = sorted(
        kernel_rows,
        key=lambda r: r["l2_drift_from_ema_init"],
        reverse=True,
    )

    _write_csv(tables / "cnn_kernel_summary.csv", kernel_rows_sorted)

    plt.figure(figsize=(8, 4))
    plt.plot(spans, drift_by_span, marker="o")
    plt.xlabel("span")
    plt.ylabel("mean L2 drift from EMA init")
    plt.title("CNN kernel drift from EMA initialization")
    plt.tight_layout()
    plt.savefig(figs / "cnn_kernel_drift_by_span.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(spans, cosine_by_span, marker="o")
    plt.xlabel("span")
    plt.ylabel("mean cosine")
    plt.title("CNN kernel cosine similarity to EMA initialization")
    plt.tight_layout()
    plt.savefig(figs / "cnn_kernel_cos_to_ema_by_span.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(spans, center_lag_by_span, marker="o")
    plt.xlabel("span")
    plt.ylabel("mean abs-center lag")
    plt.title("CNN kernel effective lag by span")
    plt.tight_layout()
    plt.savefig(figs / "cnn_kernel_effective_lag_by_span.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(spans, neg_ratio_by_span, marker="o")
    plt.xlabel("span")
    plt.ylabel("mean negative mass ratio")
    plt.title("CNN kernel negative mass ratio by span")
    plt.tight_layout()
    plt.savefig(figs / "cnn_kernel_neg_mass_by_span.png", dpi=180)
    plt.close()

    example_keys = [
        ("top_drift", "l2_drift_from_ema_init"),
        ("top_neg_mass", "neg_mass_ratio"),
        ("top_highpass", "highpass_score"),
    ]

    for prefix, key in example_keys:
        examples = sorted(kernel_rows, key=lambda r: r[key], reverse=True)[:min(top_k, 20)]

        for rank, row in enumerate(examples):
            span = int(row["span"])
            span_idx = spans.index(span)
            feature_idx = int(row["feature_index"])
            feature = str(row["feature"])

            conv = bank.convs[span_idx]
            k = conv.weight.detach().float().cpu().numpy()[feature_idx, 0, :]
            init = _ema_init_kernel(span)

            safe_feature = (
                feature.replace("/", "_")
                .replace(" ", "_")
                .replace(":", "_")
                .replace("@", "_")
                .replace("[", "_")
                .replace("]", "_")
            )

            _save_kernel_plot(
                kernel_figs / f"{prefix}_rank{rank:02d}_span{span}_{safe_feature}.png",
                k,
                init,
                title=f"{prefix}: {feature}, span={span}, {key}={row[key]:.4g}",
            )

    # ------------------------------------------------------------
    # 2. Projection usage by branch
    # ------------------------------------------------------------

    branch_rows: list[dict[str, Any]] = []

    col_offset = 0

    if include_raw_numeric:
        raw_W = W[:, col_offset:col_offset + num_features]
        raw_l2_by_feature = np.linalg.norm(raw_W, axis=0)
        raw_l2_total = float(np.linalg.norm(raw_W))
        raw_l1_total = float(np.abs(raw_W).sum())

        branch_rows.append({
            "branch": "raw_numeric",
            "span": "",
            "start_col": col_offset,
            "end_col": col_offset + num_features,
            "l2_total": raw_l2_total,
            "l1_total": raw_l1_total,
            "mean_feature_l2": float(raw_l2_by_feature.mean()),
            "max_feature_l2": float(raw_l2_by_feature.max()),
        })

        col_offset += num_features
    else:
        raw_l2_by_feature = np.zeros(num_features, dtype=np.float64)

    span_feature_importance = np.zeros((n_spans, num_features), dtype=np.float64)

    for span_idx, span in enumerate(spans):
        start = col_offset + span_idx * num_features
        end = start + num_features

        span_W = W[:, start:end]
        span_l2_by_feature = np.linalg.norm(span_W, axis=0)

        span_feature_importance[span_idx] = span_l2_by_feature

        branch_rows.append({
            "branch": "ema_cnn",
            "span": span,
            "start_col": start,
            "end_col": end,
            "l2_total": float(np.linalg.norm(span_W)),
            "l1_total": float(np.abs(span_W).sum()),
            "mean_feature_l2": float(span_l2_by_feature.mean()),
            "max_feature_l2": float(span_l2_by_feature.max()),
        })

    used_cols = raw_dim + ema_dim

    if projection_input_dim > used_cols:
        extra_W = W[:, used_cols:]
        extra_l2_cols = np.linalg.norm(extra_W, axis=0) if extra_W.shape[1] > 0 else np.zeros(0)

        branch_rows.append({
            "branch": "categorical_or_extra",
            "span": "",
            "start_col": used_cols,
            "end_col": projection_input_dim,
            "l2_total": float(np.linalg.norm(extra_W)),
            "l1_total": float(np.abs(extra_W).sum()),
            "mean_feature_l2": float(extra_l2_cols.mean()) if extra_l2_cols.size else 0.0,
            "max_feature_l2": float(extra_l2_cols.max()) if extra_l2_cols.size else 0.0,
        })

    _write_csv(tables / "cnn_branch_projection_importance.csv", branch_rows)

    branch_labels = [
        f"{r['branch']}" if r["span"] == "" else f"{r['branch']}_span{r['span']}"
        for r in branch_rows
    ]
    branch_values = np.asarray([r["l2_total"] for r in branch_rows], dtype=np.float64)

    _save_barh(
        figs / "cnn_branch_projection_importance.png",
        branch_labels,
        branch_values,
        title="Projection loading norm by input branch",
        xlabel="||W_branch||2",
        top_k=len(branch_labels),
    )

    span_rows = []

    for span_idx, span in enumerate(spans):
        vals = span_feature_importance[span_idx]

        top_idx = np.argsort(vals)[::-1][:20]

        span_rows.append({
            "span": span,
            "l2_total": float(np.linalg.norm(vals)),
            "mean_feature_l2": float(vals.mean()),
            "median_feature_l2": float(np.median(vals)),
            "max_feature_l2": float(vals.max()),
            "top_features": "|".join(feature_names[i] for i in top_idx),
            "top_feature_l2": "|".join(f"{vals[i]:.6g}" for i in top_idx),
        })

    _write_csv(tables / "cnn_span_importance.csv", span_rows)

    plt.figure(figsize=(8, 4))
    plt.bar([str(s) for s in spans], [r["l2_total"] for r in span_rows])
    plt.xlabel("span")
    plt.ylabel("L2 norm over projection columns")
    plt.title("EMA/CNN span importance in input_projection")
    plt.tight_layout()
    plt.savefig(figs / "cnn_span_projection_importance.png", dpi=180)
    plt.close()

    # ------------------------------------------------------------
    # 3. Per-feature temporal profile
    # ------------------------------------------------------------

    temporal_rows = []

    total_temporal_importance = raw_l2_by_feature + span_feature_importance.sum(axis=0)
    top_feature_idx = np.argsort(total_temporal_importance)[::-1][:min(top_k, num_features)]

    for j, feature in enumerate(feature_names):
        row: dict[str, Any] = {
            "feature": feature,
            "feature_index": j,
            "raw_l2": float(raw_l2_by_feature[j]),
            "total_ema_l2": float(span_feature_importance[:, j].sum()),
            "total_temporal_l2_sum": float(total_temporal_importance[j]),
        }

        if n_spans > 0:
            best_span_idx = int(np.argmax(span_feature_importance[:, j]))
            row["best_span"] = spans[best_span_idx]
            row["best_span_l2"] = float(span_feature_importance[best_span_idx, j])
        else:
            row["best_span"] = ""
            row["best_span_l2"] = 0.0

        for span_idx, span in enumerate(spans):
            row[f"span_{span}_l2"] = float(span_feature_importance[span_idx, j])

        temporal_rows.append(row)

    temporal_rows = sorted(
        temporal_rows,
        key=lambda r: r["total_temporal_l2_sum"],
        reverse=True,
    )

    _write_csv(tables / "cnn_feature_temporal_importance.csv", temporal_rows)

    heat = []

    for j in top_feature_idx:
        vals = []
        if include_raw_numeric:
            vals.append(raw_l2_by_feature[j])
        vals.extend(span_feature_importance[:, j].tolist())
        heat.append(vals)

    heat = np.asarray(heat, dtype=np.float64)

    col_labels = []
    if include_raw_numeric:
        col_labels.append("raw")
    col_labels.extend([f"span{s}" for s in spans])

    row_labels = [feature_names[j] for j in top_feature_idx]

    _save_heatmap(
        figs / "cnn_feature_span_importance_heatmap_top.png",
        heat,
        title="Top feature temporal-branch importance",
        xlabel="branch / span",
        ylabel="feature",
        xticklabels=col_labels,
        yticklabels=row_labels,
    )

    return {
        "status": "ok",
        "spans": spans,
        "num_features": num_features,
        "include_raw_numeric": include_raw_numeric,
        "projection_input_dim": projection_input_dim,
        "expected_raw_dim": raw_dim,
        "expected_ema_dim": ema_dim,
        "categorical_or_extra_dim": int(max(0, projection_input_dim - raw_dim - ema_dim)),
        "mean_l2_drift_by_span": {
            str(span): drift
            for span, drift in zip(spans, drift_by_span)
        },
        "mean_cos_to_ema_init_by_span": {
            str(span): cos
            for span, cos in zip(spans, cosine_by_span)
        },
        "mean_effective_lag_by_span": {
            str(span): lag
            for span, lag in zip(spans, center_lag_by_span)
        },
        "mean_neg_mass_ratio_by_span": {
            str(span): neg
            for span, neg in zip(spans, neg_ratio_by_span)
        },
        "top_kernel_drifts": kernel_rows_sorted[:20],
        "branch_projection_importance": branch_rows,
        "top_temporal_features": temporal_rows[:20],
    }


# ============================================================
# Forward activation / metric analysis
# ============================================================

def collect_forward_stats(
    *,
    model: torch.nn.Module,
    loader: Any,
    feature_stats: Mapping[str, Mapping[str, list[float]]],
    numeric_groups: list[str],
    feature_names: list[str],
    output_dir: Path,
    device: torch.device,
    max_batches: int,
    normalize_features: bool,
    target_key: str,
    target_index: int,
    pred_key: str | None,
    target_stats: Mapping[str, Mapping[str, float]],
    target_normalization: str,
    pca_sample_size: int,
    ignore_padding_mask: bool,
) -> dict[str, Any]:
    figs = output_dir / "figures"
    tables = output_dir / "tables"
    arrays = output_dir / "arrays"

    for p in [figs, tables, arrays]:
        p.mkdir(parents=True, exist_ok=True)

    model.eval()

    projection = _get_input_projection(model)

    captured: dict[str, torch.Tensor] = {}

    def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        captured["projection_input"] = inputs[0].detach()
        captured["encoder_output"] = output.detach()

    handle = projection.register_forward_hook(hook)

    projection_input_dim = int(projection.weight.shape[1])
    hidden_dim = int(projection.weight.shape[0])

    n_tokens = 0

    sum_abs_proj_in = np.zeros(projection_input_dim, dtype=np.float64)
    sum_proj_in = np.zeros(projection_input_dim, dtype=np.float64)
    sum_proj_in2 = np.zeros(projection_input_dim, dtype=np.float64)

    sum_abs_z = np.zeros(hidden_dim, dtype=np.float64)
    sum_z = np.zeros(hidden_dim, dtype=np.float64)
    sum_z2 = np.zeros(hidden_dim, dtype=np.float64)

    pred_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    weight_chunks: list[np.ndarray] = []

    z_sample_chunks: list[np.ndarray] = []
    color_sample_chunks: list[np.ndarray] = []

    rng = np.random.default_rng(123)

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= max_batches:
                    break

                batch = _to_device(batch, device)

                if normalize_features:
                    batch = _normalize_batch_features(batch, feature_stats, numeric_groups, device)

                captured.clear()
                out = model(batch)

                if "projection_input" not in captured or "encoder_output" not in captured:
                    raise RuntimeError("Forward hook did not capture projection input/output.")

                proj_in = captured["projection_input"]
                z = captured["encoder_output"]

                flat_proj_in = _flatten_last_dim(proj_in)
                flat_z = _flatten_last_dim(z)

                valid = None if ignore_padding_mask else _valid_flat_mask_from_padding(batch, proj_in)

                if valid is not None:
                    flat_proj_in = flat_proj_in[valid]
                    flat_z = flat_z[valid]

                proj_np = _detach_to_cpu_numpy(flat_proj_in)
                z_np = _detach_to_cpu_numpy(flat_z)

                n = proj_np.shape[0]
                n_tokens += n

                sum_abs_proj_in += np.abs(proj_np).sum(axis=0)
                sum_proj_in += proj_np.sum(axis=0)
                sum_proj_in2 += (proj_np ** 2).sum(axis=0)

                sum_abs_z += np.abs(z_np).sum(axis=0)
                sum_z += z_np.sum(axis=0)
                sum_z2 += (z_np ** 2).sum(axis=0)

                current_sample_size = sum(arr.shape[0] for arr in z_sample_chunks)
                remaining = max(0, pca_sample_size - current_sample_size)

                if remaining > 0:
                    take = min(remaining, n)
                    idx = rng.choice(n, size=take, replace=False) if n > take else np.arange(n)

                    z_sample_chunks.append(z_np[idx])

                    cat = batch.get("features", {}).get("categorical")

                    if torch.is_tensor(cat):
                        flat_cat = cat.reshape(-1, cat.shape[-1])

                        if valid is not None and valid.numel() == flat_cat.shape[0]:
                            flat_cat = flat_cat[valid]

                        cat_np = _detach_to_cpu_numpy(flat_cat)
                        color_sample_chunks.append(cat_np[idx, 0])
                    else:
                        color_sample_chunks.append(np.zeros(take, dtype=np.float32))

                pred = _extract_pred(out, pred_key, target_key).float()
                target = _extract_target(batch, target_key, target_index)

                if target is not None:
                    if target_normalization == "train_stats":
                        stats = target_stats.get(target_key, {"mean": 0.0, "std": 1.0})
                        mean = float(stats.get("mean", 0.0))
                        std = max(float(stats.get("std", 1.0)), 1e-12)
                        target = (target - mean) / std

                    weight = _extract_weight(batch, pred)

                    pred_flat, target_flat, weight_flat = _masked_flatten_pred_target_weight(
                        batch,
                        pred,
                        target,
                        weight,
                        ignore_padding_mask=ignore_padding_mask,
                    )

                    pred_chunks.append(_detach_to_cpu_numpy(pred_flat))
                    target_chunks.append(_detach_to_cpu_numpy(target_flat))
                    weight_chunks.append(_detach_to_cpu_numpy(weight_flat))

    finally:
        handle.remove()

    if n_tokens == 0:
        raise RuntimeError("No valid tokens collected from dataloader.")

    mean_abs_proj_in = sum_abs_proj_in / n_tokens
    mean_proj_in = sum_proj_in / n_tokens
    std_proj_in = np.sqrt(np.maximum(sum_proj_in2 / n_tokens - mean_proj_in ** 2, 0.0))

    mean_abs_z = sum_abs_z / n_tokens
    mean_z = sum_z / n_tokens
    std_z = np.sqrt(np.maximum(sum_z2 / n_tokens - mean_z ** 2, 0.0))

    W = projection.weight.detach().float().cpu().numpy()
    projection_col_l2 = np.linalg.norm(W, axis=0)

    contribution_proxy = mean_abs_proj_in * projection_col_l2

    projection_input_rows = []

    for j in range(projection_input_dim):
        projection_input_rows.append({
            "projection_input_col": j,
            "mean": float(mean_proj_in[j]),
            "std": float(std_proj_in[j]),
            "mean_abs": float(mean_abs_proj_in[j]),
            "projection_col_l2": float(projection_col_l2[j]),
            "mean_abs_times_projection_col_l2": float(contribution_proxy[j]),
        })

    projection_input_rows = sorted(
        projection_input_rows,
        key=lambda r: r["mean_abs_times_projection_col_l2"],
        reverse=True,
    )

    _write_csv(tables / "projection_input_activation_contribution_proxy.csv", projection_input_rows)

    hidden_activation_rows = []

    for d in range(hidden_dim):
        hidden_activation_rows.append({
            "hidden_dim": d,
            "mean": float(mean_z[d]),
            "std": float(std_z[d]),
            "mean_abs": float(mean_abs_z[d]),
        })

    _write_csv(tables / "hidden_activation_summary.csv", hidden_activation_rows)

    labels = [str(i) for i in range(projection_input_dim)]

    _save_barh(
        figs / "top_projection_input_activation_contribution_proxy.png",
        labels,
        contribution_proxy,
        title="Top projection input cols by mean(|x_col|) * ||W[:,col]||2",
        xlabel="activation-weight contribution proxy",
        top_k=min(50, projection_input_dim),
    )

    _save_line(
        figs / "encoder_output_mean_abs_by_hidden_dim.png",
        mean_abs_z,
        title="Encoder output mean absolute activation",
        xlabel="hidden dim",
        ylabel="mean(|z_d|)",
    )

    _save_line(
        figs / "encoder_output_std_by_hidden_dim.png",
        std_z,
        title="Encoder output std by hidden dim",
        xlabel="hidden dim",
        ylabel="std(z_d)",
    )

    metrics: dict[str, float] = {}

    if pred_chunks and target_chunks:
        pred_all = np.concatenate(pred_chunks)
        target_all = np.concatenate(target_chunks)
        weight_all = np.concatenate(weight_chunks)

        metrics = _weighted_metrics(pred_all, target_all, weight_all)

        np.save(arrays / "pred_sample.npy", pred_all[:min(len(pred_all), 200_000)])
        np.save(arrays / "target_sample.npy", target_all[:min(len(target_all), 200_000)])
        np.save(arrays / "weight_sample.npy", weight_all[:min(len(weight_all), 200_000)])

        show_n = min(50_000, len(pred_all))
        idx = rng.choice(len(pred_all), size=show_n, replace=False) if len(pred_all) > show_n else np.arange(len(pred_all))

        plt.figure(figsize=(6, 6))
        plt.scatter(target_all[idx], pred_all[idx], s=2, alpha=0.25)
        plt.xlabel("target")
        plt.ylabel("prediction")
        plt.title(
            "Prediction vs target\n"
            f"cos={metrics.get('weighted_cosine', float('nan')):.4g}, "
            f"R2={metrics.get('weighted_r2_zero_benchmark', float('nan')):.4g}"
        )
        plt.tight_layout()
        plt.savefig(figs / "pred_vs_target_scatter.png", dpi=180)
        plt.close()

    z_pca_explained = None

    if z_sample_chunks:
        z_sample = np.concatenate(z_sample_chunks, axis=0)
        z_coords, z_explained = _pca_2d(z_sample)
        z_pca_explained = z_explained.tolist()

        color = np.concatenate(color_sample_chunks, axis=0) if color_sample_chunks else None

        np.save(arrays / "encoder_output_pca_sample.npy", z_sample)

        _save_scatter(
            figs / "encoder_output_pca_sample.png",
            z_coords,
            title=f"Encoder output PCA sample; explained={z_explained[0]:.2%},{z_explained[1]:.2%}",
            color=color,
        )

    return {
        "n_tokens": int(n_tokens),
        "metrics": metrics,
        "encoder_output_pca_explained": z_pca_explained,
        "top_projection_input_cols_by_activation_contribution_proxy": projection_input_rows[:20],
        "top_hidden_by_mean_abs_activation": sorted(
            hidden_activation_rows,
            key=lambda r: r["mean_abs"],
            reverse=True,
        )[:20],
    }


# ============================================================
# Gradient attribution
# ============================================================

def collect_gradient_attribution(
    *,
    model: torch.nn.Module,
    loader: Any,
    feature_stats: Mapping[str, Mapping[str, list[float]]],
    numeric_groups: list[str],
    feature_names: list[str],
    output_dir: Path,
    device: torch.device,
    max_batches: int,
    normalize_features: bool,
    target_key: str,
    pred_key: str | None,
    ignore_padding_mask: bool,
) -> dict[str, Any]:
    figs = output_dir / "figures"
    tables = output_dir / "tables"

    for p in [figs, tables]:
        p.mkdir(parents=True, exist_ok=True)

    model.eval()

    num_features = len(feature_names)

    sum_abs_grad_x = np.zeros(num_features, dtype=np.float64)
    sum_abs_grad = np.zeros(num_features, dtype=np.float64)

    n_tokens = 0

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break

        batch = _to_device(batch, device)

        if normalize_features:
            batch = _normalize_batch_features(batch, feature_stats, numeric_groups, device)

        for group in numeric_groups:
            x = batch["features"].get(group)

            if x is not None:
                batch["features"][group] = x.detach().clone().requires_grad_(True)

        model.zero_grad(set_to_none=True)

        out = model(batch)
        pred = _extract_pred(out, pred_key, target_key).float()

        valid = None if ignore_padding_mask else _valid_flat_mask_from_padding(batch, pred)

        if valid is not None and pred.ndim >= 3 and pred.shape[-1] == 1:
            score = pred.reshape(-1)[valid].sum()
        elif valid is not None and pred.ndim == 2:
            score = pred.reshape(-1)[valid].sum()
        else:
            score = pred.sum()

        score.backward()

        xs = []
        gs = []

        for group in numeric_groups:
            x = batch["features"].get(group)

            if x is None:
                continue

            if x.grad is None:
                continue

            xs.append(x.detach())
            gs.append(x.grad.detach())

        if not xs:
            continue

        x_all = torch.cat(xs, dim=-1)
        g_all = torch.cat(gs, dim=-1)

        flat_x = _flatten_last_dim(x_all)
        flat_g = _flatten_last_dim(g_all)

        valid_x = None if ignore_padding_mask else _valid_flat_mask_from_padding(batch, x_all)

        if valid_x is not None:
            flat_x = flat_x[valid_x]
            flat_g = flat_g[valid_x]

        n = flat_x.shape[0]
        n_tokens += n

        gx = _detach_to_cpu_numpy(flat_g * flat_x)
        gg = _detach_to_cpu_numpy(flat_g)

        sum_abs_grad_x += np.abs(gx).sum(axis=0)
        sum_abs_grad += np.abs(gg).sum(axis=0)

    if n_tokens == 0:
        return {"status": "skipped_no_gradients"}

    mean_abs_grad_x = sum_abs_grad_x / n_tokens
    mean_abs_grad = sum_abs_grad / n_tokens

    rows = []

    for j, name in enumerate(feature_names):
        rows.append({
            "feature": name,
            "feature_index": j,
            "mean_abs_grad_x_input": float(mean_abs_grad_x[j]),
            "mean_abs_grad": float(mean_abs_grad[j]),
        })

    rows = sorted(rows, key=lambda r: r["mean_abs_grad_x_input"], reverse=True)

    _write_csv(tables / "feature_gradient_x_input_attribution.csv", rows)

    _save_barh(
        figs / "top_feature_gradient_x_input.png",
        feature_names,
        mean_abs_grad_x,
        title="Top raw features by mean(|gradient * input|)",
        xlabel="mean(|grad * input|)",
        top_k=min(50, len(feature_names)),
    )

    return {
        "n_tokens": int(n_tokens),
        "top_features_by_grad_x_input": rows[:20],
    }


# ============================================================
# Single-feature raw input ablation
# ============================================================

def run_ablation(
    *,
    model: torch.nn.Module,
    loader_builder: Any,
    feature_stats: Mapping[str, Mapping[str, list[float]]],
    numeric_groups: list[str],
    feature_names: list[str],
    output_dir: Path,
    device: torch.device,
    max_batches: int,
    normalize_features: bool,
    target_key: str,
    target_index: int,
    pred_key: str | None,
    target_stats: Mapping[str, Mapping[str, float]],
    target_normalization: str,
    ablate_top_features: list[str],
    ignore_padding_mask: bool,
) -> dict[str, Any]:
    tables = output_dir / "tables"
    figs = output_dir / "figures"

    for p in [tables, figs]:
        p.mkdir(parents=True, exist_ok=True)

    feature_to_idx = {name: i for i, name in enumerate(feature_names)}
    ablate_indices = [feature_to_idx[name] for name in ablate_top_features if name in feature_to_idx]

    def eval_with_ablation(ablate_idx: int | None) -> dict[str, float]:
        loader = loader_builder()

        model.eval()

        pred_chunks: list[np.ndarray] = []
        target_chunks: list[np.ndarray] = []
        weight_chunks: list[np.ndarray] = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= max_batches:
                    break

                batch = _to_device(batch, device)

                if normalize_features:
                    batch = _normalize_batch_features(batch, feature_stats, numeric_groups, device)

                if ablate_idx is not None:
                    offset = 0

                    for group in numeric_groups:
                        x = batch["features"].get(group)

                        if x is None:
                            continue

                        width = x.shape[-1]

                        if offset <= ablate_idx < offset + width:
                            local_idx = ablate_idx - offset
                            x = x.clone()
                            x[..., local_idx] = 0.0
                            batch["features"][group] = x
                            break

                        offset += width

                out = model(batch)

                pred = _extract_pred(out, pred_key, target_key).float()
                target = _extract_target(batch, target_key, target_index)

                if target is None:
                    continue

                if target_normalization == "train_stats":
                    stats = target_stats.get(target_key, {"mean": 0.0, "std": 1.0})
                    mean = float(stats.get("mean", 0.0))
                    std = max(float(stats.get("std", 1.0)), 1e-12)
                    target = (target - mean) / std

                weight = _extract_weight(batch, pred)

                pred_flat, target_flat, weight_flat = _masked_flatten_pred_target_weight(
                    batch,
                    pred,
                    target,
                    weight,
                    ignore_padding_mask=ignore_padding_mask,
                )

                pred_chunks.append(_detach_to_cpu_numpy(pred_flat))
                target_chunks.append(_detach_to_cpu_numpy(target_flat))
                weight_chunks.append(_detach_to_cpu_numpy(weight_flat))

        if not pred_chunks:
            return {
                "mse": float("nan"),
                "weighted_r2_zero_benchmark": float("nan"),
                "weighted_cosine": float("nan"),
                "corr": float("nan"),
                "n": 0.0,
            }

        return _weighted_metrics(
            np.concatenate(pred_chunks),
            np.concatenate(target_chunks),
            np.concatenate(weight_chunks),
        )

    baseline = eval_with_ablation(None)

    rows = []

    for idx in ablate_indices:
        feature = feature_names[idx]
        metrics = eval_with_ablation(idx)

        row = {
            "feature": feature,
            "feature_index": idx,
            "baseline_weighted_cosine": baseline.get("weighted_cosine", float("nan")),
            "ablated_weighted_cosine": metrics.get("weighted_cosine", float("nan")),
            "delta_weighted_cosine": baseline.get("weighted_cosine", float("nan")) - metrics.get("weighted_cosine", float("nan")),
            "baseline_r2": baseline.get("weighted_r2_zero_benchmark", float("nan")),
            "ablated_r2": metrics.get("weighted_r2_zero_benchmark", float("nan")),
            "delta_r2": baseline.get("weighted_r2_zero_benchmark", float("nan")) - metrics.get("weighted_r2_zero_benchmark", float("nan")),
            "baseline_mse": baseline.get("mse", float("nan")),
            "ablated_mse": metrics.get("mse", float("nan")),
            "delta_mse": metrics.get("mse", float("nan")) - baseline.get("mse", float("nan")),
        }

        rows.append(row)

    rows = sorted(rows, key=lambda r: abs(r["delta_weighted_cosine"]), reverse=True)

    _write_csv(tables / "single_feature_zero_ablation.csv", rows)

    if rows:
        labels = [r["feature"] for r in rows]
        values = np.asarray([r["delta_weighted_cosine"] for r in rows])

        _save_barh(
            figs / "single_feature_ablation_delta_cosine.png",
            labels,
            values,
            title="Single-feature zero ablation: baseline cosine - ablated cosine",
            xlabel="delta weighted cosine",
            top_k=min(50, len(labels)),
        )

    return {
        "baseline": baseline,
        "num_ablated_features": len(rows),
        "top_ablation_results": rows[:20],
    }


# ============================================================
# CNN span ablation
# ============================================================

def run_cnn_span_ablation(
    *,
    model: torch.nn.Module,
    loader_builder: Any,
    feature_stats: Mapping[str, Mapping[str, list[float]]],
    numeric_groups: list[str],
    output_dir: Path,
    device: torch.device,
    max_batches: int,
    normalize_features: bool,
    target_key: str,
    target_index: int,
    pred_key: str | None,
    target_stats: Mapping[str, Mapping[str, float]],
    target_normalization: str,
    ignore_padding_mask: bool,
) -> dict[str, Any]:
    if not _has_ema_bank(model):
        return {"status": "skipped_no_causal_ema_bank"}

    tables = output_dir / "tables"
    figs = output_dir / "figures"

    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    bank = _get_ema_bank(model)
    assert bank is not None

    spans = [int(s) for s in bank.spans]

    def eval_with_span_mask(mask_span_idx: int | None) -> dict[str, float]:
        loader = loader_builder()
        model.eval()

        pred_chunks: list[np.ndarray] = []
        target_chunks: list[np.ndarray] = []
        weight_chunks: list[np.ndarray] = []

        handles = []

        if mask_span_idx is not None:
            conv = bank.convs[mask_span_idx]

            def zero_hook(_module, _inputs, output):
                return torch.zeros_like(output)

            handles.append(conv.register_forward_hook(zero_hook))

        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(loader):
                    if batch_idx >= max_batches:
                        break

                    batch = _to_device(batch, device)

                    if normalize_features:
                        batch = _normalize_batch_features(batch, feature_stats, numeric_groups, device)

                    out = model(batch)

                    pred = _extract_pred(out, pred_key, target_key).float()
                    target = _extract_target(batch, target_key, target_index)

                    if target is None:
                        continue

                    if target_normalization == "train_stats":
                        stats = target_stats.get(target_key, {"mean": 0.0, "std": 1.0})
                        mean = float(stats.get("mean", 0.0))
                        std = max(float(stats.get("std", 1.0)), 1e-12)
                        target = (target - mean) / std

                    weight = _extract_weight(batch, pred)

                    pred_flat, target_flat, weight_flat = _masked_flatten_pred_target_weight(
                        batch,
                        pred,
                        target,
                        weight,
                        ignore_padding_mask=ignore_padding_mask,
                    )

                    pred_chunks.append(_detach_to_cpu_numpy(pred_flat))
                    target_chunks.append(_detach_to_cpu_numpy(target_flat))
                    weight_chunks.append(_detach_to_cpu_numpy(weight_flat))

        finally:
            for h in handles:
                h.remove()

        if not pred_chunks:
            return {
                "mse": float("nan"),
                "weighted_r2_zero_benchmark": float("nan"),
                "weighted_cosine": float("nan"),
                "corr": float("nan"),
                "n": 0.0,
            }

        return _weighted_metrics(
            np.concatenate(pred_chunks),
            np.concatenate(target_chunks),
            np.concatenate(weight_chunks),
        )

    baseline = eval_with_span_mask(None)

    rows = []

    for span_idx, span in enumerate(spans):
        metrics = eval_with_span_mask(span_idx)

        rows.append({
            "span": span,
            "baseline_weighted_cosine": baseline.get("weighted_cosine", float("nan")),
            "ablated_weighted_cosine": metrics.get("weighted_cosine", float("nan")),
            "delta_weighted_cosine": baseline.get("weighted_cosine", float("nan")) - metrics.get("weighted_cosine", float("nan")),
            "baseline_r2": baseline.get("weighted_r2_zero_benchmark", float("nan")),
            "ablated_r2": metrics.get("weighted_r2_zero_benchmark", float("nan")),
            "delta_r2": baseline.get("weighted_r2_zero_benchmark", float("nan")) - metrics.get("weighted_r2_zero_benchmark", float("nan")),
            "baseline_mse": baseline.get("mse", float("nan")),
            "ablated_mse": metrics.get("mse", float("nan")),
            "delta_mse": metrics.get("mse", float("nan")) - baseline.get("mse", float("nan")),
        })

    rows = sorted(rows, key=lambda r: abs(r["delta_weighted_cosine"]), reverse=True)

    _write_csv(tables / "cnn_span_ablation.csv", rows)

    if rows:
        labels = [f"span{r['span']}" for r in rows]
        values = np.asarray([r["delta_weighted_cosine"] for r in rows], dtype=np.float64)

        _save_barh(
            figs / "cnn_span_ablation_delta_cosine.png",
            labels,
            values,
            title="CNN span ablation: baseline cosine - ablated cosine",
            xlabel="delta weighted cosine",
            top_k=len(labels),
        )

    return {
        "status": "ok",
        "baseline": baseline,
        "rows": rows,
    }


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze TabularSequenceEncoder and optional causal EMA/CNN bank.")

    parser.add_argument("--config", required=True, help="Path to training config.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.")
    parser.add_argument("--output-dir", required=True, help="Directory to dump figures/tables/summary.")

    parser.add_argument("--split", choices=["train", "val", "auto"], default="val")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--grad-max-batches", type=int, default=10)
    parser.add_argument("--ablation-max-batches", type=int, default=20)

    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--n-clusters", type=int, default=12)
    parser.add_argument("--pca-sample-size", type=int, default=50_000)

    parser.add_argument(
        "--no-normalize-features",
        action="store_true",
        help="Disable feature normalization before model forward.",
    )

    parser.add_argument(
        "--target-normalization",
        choices=["train_stats", "none"],
        default="train_stats",
        help="Use train_stats if your objective standardizes target internally.",
    )

    parser.add_argument(
        "--ignore-padding-mask",
        action="store_true",
        help="Do not mask padded tokens when collecting stats / metrics.",
    )

    parser.add_argument(
        "--apply-config-vocab",
        action="store_true",
        help=(
            "Apply dataset_cfg.vocab_path before building dataloader. "
            "Keep this OFF if your current dataio already applies vocabs.pkl internally."
        ),
    )

    parser.add_argument("--skip-gradient", action="store_true")

    parser.add_argument("--run-ablation", action="store_true")

    parser.add_argument(
        "--ablation-top-n",
        type=int,
        default=30,
        help="Ablate top-N raw features selected by raw-branch weight L2. Only used with --run-ablation.",
    )

    parser.add_argument(
        "--run-cnn-span-ablation",
        action="store_true",
        help="Zero out each causal CNN span branch and evaluate metric drop.",
    )

    parser.add_argument(
        "--prefer-raw-model",
        action="store_true",
        help="If checkpoint contains EMA and raw model, load raw model instead of EMA.",
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    output_dir = Path(args.output_dir)

    figs = output_dir / "figures"
    tables = output_dir / "tables"
    arrays = output_dir / "arrays"

    for p in [output_dir, figs, tables, arrays]:
        p.mkdir(parents=True, exist_ok=True)

    cfg = load_config_with_inheritance(args.config)
    data_cfg = dict(cfg["data"])

    train_dataset_cfg, val_dataset_cfg = _resolve_dataset_cfgs(data_cfg)
    model_cfg = _inject_vocab_sizes(dict(cfg["model"]), train_dataset_cfg)

    numeric_groups = [str(v) for v in model_cfg.get("numeric_feature_groups", ["continuous"])]

    print("[1/9] Loading train frame for stats / feature names...")
    train_df = _load_frame(train_dataset_cfg, apply_config_vocab=args.apply_config_vocab)

    train_feature_cols = _resolve_feature_cols(train_df, train_dataset_cfg, model_cfg)

    feature_stats = _compute_feature_stats(
        train_df,
        train_feature_cols,
        numeric_groups=numeric_groups,
    )

    target_cols = _resolve_target_cols(train_dataset_cfg)
    target_stats = _compute_target_stats(train_df, target_cols)

    feature_names: list[str] = []

    for group in numeric_groups:
        feature_names.extend(train_feature_cols.get(group, []))

    if not feature_names:
        raise ValueError(f"No numeric features found for numeric_groups={numeric_groups}")

    if args.split == "train":
        analysis_dataset_cfg = train_dataset_cfg
        analysis_df = train_df
        loader_cfg = data_cfg.get("train_dataloader", {})
        split_used = "train"

    elif args.split == "val":
        if val_dataset_cfg is not None:
            print("[2/9] Loading val frame...")
            analysis_dataset_cfg = val_dataset_cfg
            analysis_df = _load_frame(val_dataset_cfg, apply_config_vocab=args.apply_config_vocab)
            loader_cfg = data_cfg.get("val_dataloader", {"shuffle": False})
            split_used = "val"
        else:
            analysis_dataset_cfg = train_dataset_cfg
            analysis_df = train_df
            loader_cfg = data_cfg.get("train_dataloader", {})
            split_used = "train_fallback_no_val"

    else:
        if val_dataset_cfg is not None:
            print("[2/9] Loading val frame...")
            analysis_dataset_cfg = val_dataset_cfg
            analysis_df = _load_frame(val_dataset_cfg, apply_config_vocab=args.apply_config_vocab)
            loader_cfg = data_cfg.get("val_dataloader", {"shuffle": False})
            split_used = "val"
        else:
            analysis_dataset_cfg = train_dataset_cfg
            analysis_df = train_df
            loader_cfg = data_cfg.get("train_dataloader", {})
            split_used = "train"

    def make_loader() -> Any:
        return _build_dataloader(
            analysis_df,
            analysis_dataset_cfg,
            loader_cfg,
            model_cfg,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    print("[3/9] Building analysis dataloader...")
    loader = make_loader()

    print("[4/9] Building model and initializing LazyLinear...")
    device = torch.device(args.device)

    model = build_model(model_cfg)
    model.to(device)

    first_batch = next(iter(loader))
    first_batch = _to_device(first_batch, device)

    if not args.no_normalize_features:
        first_batch = _normalize_batch_features(first_batch, feature_stats, numeric_groups, device)

    with torch.no_grad():
        _ = model(first_batch)

    print("[5/9] Loading checkpoint...")
    ckpt_info = _load_checkpoint(
        model,
        args.checkpoint,
        prefer_ema=not args.prefer_raw_model,
    )

    model.to(device)
    model.eval()

    target_key, target_index, pred_key = _infer_target_key_and_index(cfg, analysis_dataset_cfg)

    resolved = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "output_dir": str(output_dir),
        "split_requested": args.split,
        "split_used": split_used,
        "device": str(device),
        "numeric_groups": numeric_groups,
        "num_features": len(feature_names),
        "feature_names_head": feature_names[:20],
        "target_cols": target_cols,
        "target_key": target_key,
        "target_index": target_index,
        "pred_key": pred_key,
        "normalize_features": not args.no_normalize_features,
        "target_normalization": args.target_normalization,
        "padding_mask_semantics": "True means valid token; False means padded token.",
        "sample_weight_semantics": "uniform ones; current dataio has no weight field.",
        "has_causal_ema_bank": _has_ema_bank(model),
        "apply_config_vocab": args.apply_config_vocab,
        "checkpoint_info": ckpt_info,
    }

    (output_dir / "analysis_config.json").write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[6/9] Analyzing input projection weights / clusters...")
    projection_summary = analyze_projection(
        model=model,
        feature_names=feature_names,
        output_dir=output_dir,
        top_k=args.top_k,
        n_clusters=args.n_clusters,
        seed=args.seed,
    )

    print("[7/9] Analyzing causal EMA/CNN bank if present...")
    cnn_summary = analyze_causal_ema_bank(
        model=model,
        feature_names=feature_names,
        output_dir=output_dir,
        top_k=args.top_k,
    )

    print("[8/9] Collecting forward activation / prediction stats...")
    loader = make_loader()

    forward_summary = collect_forward_stats(
        model=model,
        loader=loader,
        feature_stats=feature_stats,
        numeric_groups=numeric_groups,
        feature_names=feature_names,
        output_dir=output_dir,
        device=device,
        max_batches=args.max_batches,
        normalize_features=not args.no_normalize_features,
        target_key=target_key,
        target_index=target_index,
        pred_key=pred_key,
        target_stats=target_stats,
        target_normalization=args.target_normalization,
        pca_sample_size=args.pca_sample_size,
        ignore_padding_mask=args.ignore_padding_mask,
    )

    gradient_summary: dict[str, Any] = {"status": "skipped"}

    if not args.skip_gradient:
        print("[9/9] Collecting gradient x input attribution...")
        loader = make_loader()

        gradient_summary = collect_gradient_attribution(
            model=model,
            loader=loader,
            feature_stats=feature_stats,
            numeric_groups=numeric_groups,
            feature_names=feature_names,
            output_dir=output_dir,
            device=device,
            max_batches=args.grad_max_batches,
            normalize_features=not args.no_normalize_features,
            target_key=target_key,
            pred_key=pred_key,
            ignore_padding_mask=args.ignore_padding_mask,
        )
    else:
        print("[9/9] Gradient attribution skipped.")

    ablation_summary: dict[str, Any] = {"status": "skipped"}

    if args.run_ablation:
        print("[extra] Running single-feature raw input zero ablation...")

        full_rows = []
        feature_importance_path = tables / "feature_raw_branch_weight_importance.csv"

        with feature_importance_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            full_rows = list(reader)

        top_features = [r["feature"] for r in full_rows[:args.ablation_top_n]]

        ablation_summary = run_ablation(
            model=model,
            loader_builder=make_loader,
            feature_stats=feature_stats,
            numeric_groups=numeric_groups,
            feature_names=feature_names,
            output_dir=output_dir,
            device=device,
            max_batches=args.ablation_max_batches,
            normalize_features=not args.no_normalize_features,
            target_key=target_key,
            target_index=target_index,
            pred_key=pred_key,
            target_stats=target_stats,
            target_normalization=args.target_normalization,
            ablate_top_features=top_features,
            ignore_padding_mask=args.ignore_padding_mask,
        )

    cnn_span_ablation_summary: dict[str, Any] = {"status": "skipped"}

    if args.run_cnn_span_ablation:
        print("[extra] Running CNN span ablation...")

        cnn_span_ablation_summary = run_cnn_span_ablation(
            model=model,
            loader_builder=make_loader,
            feature_stats=feature_stats,
            numeric_groups=numeric_groups,
            output_dir=output_dir,
            device=device,
            max_batches=args.ablation_max_batches,
            normalize_features=not args.no_normalize_features,
            target_key=target_key,
            target_index=target_index,
            pred_key=pred_key,
            target_stats=target_stats,
            target_normalization=args.target_normalization,
            ignore_padding_mask=args.ignore_padding_mask,
        )

    summary = {
        "resolved": resolved,
        "projection": projection_summary,
        "cnn_bank": cnn_summary,
        "forward": forward_summary,
        "gradient": gradient_summary,
        "ablation": ablation_summary,
        "cnn_span_ablation": cnn_span_ablation_summary,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md: list[str] = []

    md.append("# TabularSequenceEncoder Analysis\n\n")

    md.append("## Model abstraction\n\n")

    if cnn_summary.get("status") == "ok":
        md.append("Current encoder is interpreted as:\n\n")
        md.append("```text\n")
        md.append("raw:       x_numeric\n")
        md.append("cnn bank:  causal_ema_bank(x_numeric)\n")
        md.append("cat:       categorical embeddings, if configured\n")
        md.append("z:         input_projection(concat(raw, cnn bank, cat))\n")
        md.append("```\n\n")
    else:
        md.append("Current encoder is interpreted as:\n\n")
        md.append("```text\n")
        md.append("z = W x + b\n")
        md.append("```\n\n")

    md.append("## Run information\n\n")
    md.append(f"- split used: `{split_used}`\n")
    md.append(f"- checkpoint key loaded: `{ckpt_info.get('used_key')}`\n")
    md.append(f"- number of numeric features: `{len(feature_names)}`\n")
    md.append(f"- input projection shape: `{projection_summary['weight_shape']}`\n")
    md.append(f"- normalize features: `{not args.no_normalize_features}`\n")
    md.append(f"- target normalization for metrics: `{args.target_normalization}`\n")
    md.append("- padding mask semantics: `True = valid token`, `False = padded token`\n")
    md.append("- sample weights: uniform ones, because current dataio has no weight field\n")
    md.append(f"- has causal EMA/CNN bank: `{_has_ema_bank(model)}`\n\n")

    md.append("## Main figures\n\n")
    md.append("- `figures/top_feature_raw_branch_weight_l2.png`: raw branch features with largest loading norm `||W_raw[:, j]||2`.\n")
    md.append("- `figures/feature_raw_loading_pca.png`: PCA of raw feature loading vectors, if raw branch exists.\n")
    md.append("- `figures/hidden_channel_weight_l2.png`: hidden channel loading norms `||W[d, :]||2`.\n")
    md.append("- `figures/hidden_channel_pca.png`: PCA of hidden channel loading vectors `W[d, :]`.\n")
    md.append("- `figures/top_projection_input_activation_contribution_proxy.png`: `mean(|input_col|) * ||W[:, col]||2`.\n")
    md.append("- `figures/encoder_output_mean_abs_by_hidden_dim.png`: average absolute encoder output by hidden channel.\n")
    md.append("- `figures/encoder_output_std_by_hidden_dim.png`: encoder output std by hidden channel.\n")
    md.append("- `figures/encoder_output_pca_sample.png`: PCA sample of actual encoder outputs.\n")
    md.append("- `figures/pred_vs_target_scatter.png`: prediction vs target scatter, if target is available.\n")

    if cnn_summary.get("status") == "ok":
        md.append("- `figures/cnn_kernel_drift_by_span.png`: learned CNN kernels' average L2 drift from EMA initialization.\n")
        md.append("- `figures/cnn_kernel_cos_to_ema_by_span.png`: average cosine similarity to EMA initialization.\n")
        md.append("- `figures/cnn_kernel_effective_lag_by_span.png`: effective lag / center of mass of learned filters.\n")
        md.append("- `figures/cnn_kernel_neg_mass_by_span.png`: average negative mass ratio by span.\n")
        md.append("- `figures/cnn_branch_projection_importance.png`: raw vs CNN span vs categorical branch loading norm in input projection.\n")
        md.append("- `figures/cnn_span_projection_importance.png`: projection loading norm by CNN span.\n")
        md.append("- `figures/cnn_feature_span_importance_heatmap_top.png`: top features' raw/span loading profile.\n")
        md.append("- `figures/cnn_kernel_examples/`: example learned kernels with large drift / negative mass / high-pass behavior.\n")

    if not args.skip_gradient:
        md.append("- `figures/top_feature_gradient_x_input.png`: raw feature gradient × input attribution.\n")

    if args.run_ablation:
        md.append("- `figures/single_feature_ablation_delta_cosine.png`: metric drop after zeroing selected raw features.\n")

    if args.run_cnn_span_ablation:
        md.append("- `figures/cnn_span_ablation_delta_cosine.png`: metric drop after zeroing each CNN span.\n")

    md.append("\n")

    md.append("## Main tables\n\n")
    md.append("- `tables/feature_raw_branch_weight_importance.csv`\n")
    md.append("- `tables/feature_raw_loading_clusters.csv`, if raw feature PCA is available\n")
    md.append("- `tables/hidden_channel_summary.csv`\n")
    md.append("- `tables/projection_input_activation_contribution_proxy.csv`\n")
    md.append("- `tables/hidden_activation_summary.csv`\n")

    if cnn_summary.get("status") == "ok":
        md.append("- `tables/cnn_kernel_summary.csv`\n")
        md.append("- `tables/cnn_branch_projection_importance.csv`\n")
        md.append("- `tables/cnn_span_importance.csv`\n")
        md.append("- `tables/cnn_feature_temporal_importance.csv`\n")

    if not args.skip_gradient:
        md.append("- `tables/feature_gradient_x_input_attribution.csv`\n")

    if args.run_ablation:
        md.append("- `tables/single_feature_zero_ablation.csv`\n")

    if args.run_cnn_span_ablation:
        md.append("- `tables/cnn_span_ablation.csv`\n")

    md.append("\n")

    md.append("## Prediction metrics on sampled batches\n\n")
    metrics = forward_summary.get("metrics", {})

    if metrics:
        for k, v in metrics.items():
            md.append(f"- `{k}`: `{v}`\n")
    else:
        md.append("- No target/prediction metric was computed.\n")

    md.append("\n")

    md.append("## Top raw features by projection weight norm\n\n")
    for row in projection_summary["top_features_by_raw_branch_weight_l2"][:10]:
        md.append(f"- `{row['feature']}`: raw_branch_weight_l2={row['raw_branch_weight_l2']:.6g}\n")

    md.append("\n")

    if cnn_summary.get("status") == "ok":
        md.append("## CNN / EMA bank summary\n\n")
        md.append(f"- spans: `{cnn_summary['spans']}`\n")
        md.append(f"- include raw numeric: `{cnn_summary['include_raw_numeric']}`\n")
        md.append(f"- categorical or extra dim after raw+EMA: `{cnn_summary['categorical_or_extra_dim']}`\n")

        md.append("\n### Mean cosine to EMA init by span\n\n")
        for span, val in cnn_summary["mean_cos_to_ema_init_by_span"].items():
            md.append(f"- span `{span}`: `{val:.6g}`\n")

        md.append("\n### Mean effective lag by span\n\n")
        for span, val in cnn_summary["mean_effective_lag_by_span"].items():
            md.append(f"- span `{span}`: `{val:.6g}`\n")

        md.append("\n### Mean negative mass ratio by span\n\n")
        for span, val in cnn_summary["mean_neg_mass_ratio_by_span"].items():
            md.append(f"- span `{span}`: `{val:.6g}`\n")

        md.append("\n### Top temporal features\n\n")
        for row in cnn_summary["top_temporal_features"][:10]:
            md.append(
                f"- `{row['feature']}`: "
                f"raw_l2={row['raw_l2']:.6g}, "
                f"total_ema_l2={row['total_ema_l2']:.6g}, "
                f"best_span={row['best_span']}, "
                f"best_span_l2={row['best_span_l2']:.6g}\n"
            )
        md.append("\n")
    else:
        md.append("## CNN / EMA bank summary\n\n")
        md.append(f"- CNN bank analysis skipped: `{cnn_summary.get('status')}`\n\n")

    md.append("## Top projection input columns by activation-weight contribution proxy\n\n")
    for row in forward_summary["top_projection_input_cols_by_activation_contribution_proxy"][:10]:
        md.append(
            f"- col `{row['projection_input_col']}`: "
            f"mean_abs_times_projection_col_l2={row['mean_abs_times_projection_col_l2']:.6g}, "
            f"mean_abs={row['mean_abs']:.6g}, "
            f"projection_col_l2={row['projection_col_l2']:.6g}\n"
        )

    md.append("\n")

    if not args.skip_gradient and gradient_summary.get("top_features_by_grad_x_input"):
        md.append("## Top raw features by gradient × input\n\n")
        for row in gradient_summary["top_features_by_grad_x_input"][:10]:
            md.append(f"- `{row['feature']}`: grad_x_input={row['mean_abs_grad_x_input']:.6g}\n")
        md.append("\n")

    if args.run_ablation and ablation_summary.get("top_ablation_results"):
        md.append("## Top single raw-feature ablation results\n\n")
        for row in ablation_summary["top_ablation_results"][:10]:
            md.append(
                f"- `{row['feature']}`: "
                f"delta_cos={row['delta_weighted_cosine']:.6g}, "
                f"delta_r2={row['delta_r2']:.6g}, "
                f"delta_mse={row['delta_mse']:.6g}\n"
            )
        md.append("\n")

    if args.run_cnn_span_ablation and cnn_span_ablation_summary.get("rows"):
        md.append("## CNN span ablation results\n\n")
        for row in cnn_span_ablation_summary["rows"]:
            md.append(
                f"- span `{row['span']}`: "
                f"delta_cos={row['delta_weighted_cosine']:.6g}, "
                f"delta_r2={row['delta_r2']:.6g}, "
                f"delta_mse={row['delta_mse']:.6g}\n"
            )
        md.append("\n")

    md.append("## Interpretation notes\n\n")
    md.append("1. `feature_raw_branch_weight_importance.csv` only describes direct raw-feature loading. If CNN bank exists, raw loading is not the whole story.\n")
    md.append("2. `cnn_feature_temporal_importance.csv` is the best table for seeing which feature uses which temporal span.\n")
    md.append("3. `cnn_kernel_summary.csv` tells you whether each per-feature temporal filter remains EMA-like or becomes signed/high-pass/nonmonotone.\n")
    md.append("4. `cnn_branch_projection_importance.csv` tells you whether the model mostly uses raw features, EMA/CNN features, or categorical embeddings.\n")
    md.append("5. `gradient_x_input` is local sensitivity. Trust it more when it agrees with raw-feature ablation or CNN span ablation.\n")
    md.append("6. The metric code masks padded tokens by default using `padding_mask=True` as valid.\n")
    md.append("7. Since current dataio has no weight, reported weighted metrics are effectively unweighted over valid tokens.\n")

    (output_dir / "summary.md").write_text("".join(md), encoding="utf-8")

    print(f"[done] analysis dumped to: {output_dir}")


if __name__ == "__main__":
    main()