from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jump_dl.src.config import load_config_with_inheritance
from jump_dl.src.models import build_model
from jump_dl.src.utils.externals import ensure_torch

# Reuse utilities from the first analysis script.
from jump_dl.scripts.analyze_encoder import (
    _set_seed,
    _resolve_dataset_cfgs,
    _resolve_target_cols,
    _load_frame,
    _resolve_feature_cols,
    _compute_feature_stats,
    _compute_target_stats,
    _build_dataloader,
    _inject_vocab_sizes,
    _to_device,
    _normalize_batch_features,
    _extract_pred,
    _extract_target,
    _extract_weight,
    _masked_flatten_pred_target_weight,
    _weighted_metrics,
    _infer_target_key_and_index,
    _load_checkpoint,
    _get_input_projection,
    _has_ema_bank,
    _get_ema_bank,
    _write_csv,
    _save_barh,
    _save_heatmap,
)

torch = ensure_torch()


# ============================================================
# Generic plotting / math helpers
# ============================================================

def _safe_name(s: str) -> str:
    return (
        str(s)
        .replace("/", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace("@", "_")
        .replace("[", "_")
        .replace("]", "_")
        .replace("(", "_")
        .replace(")", "_")
    )


def _detach_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def _flatten_last_dim(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.shape[-1])


def _pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    u, s, _vt = np.linalg.svd(x, full_matrices=False)
    coords = u[:, :2] * s[:2]
    explained = (s[:2] ** 2) / max(float((s ** 2).sum()), 1e-12)
    return coords, explained


def _cosine_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def _save_line_plot(
    path: Path,
    x: np.ndarray,
    ys: list[tuple[str, np.ndarray]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    for label, y in ys:
        plt.plot(x, y, marker="o", label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _save_scatter(
    path: Path,
    coords: np.ndarray,
    *,
    title: str,
    labels: list[str] | None = None,
    annotate_indices: np.ndarray | None = None,
    color: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    if color is None:
        plt.scatter(coords[:, 0], coords[:, 1], s=12, alpha=0.75)
    else:
        plt.scatter(coords[:, 0], coords[:, 1], s=12, alpha=0.75, c=color)
        plt.colorbar()

    if labels is not None and annotate_indices is not None:
        for i in annotate_indices:
            plt.text(coords[i, 0], coords[i, 1], labels[i], fontsize=6)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _save_group_bar(path: Path, rows: list[dict[str, Any]], key: str, *, title: str, xlabel: str) -> None:
    labels = [str(r["group"]) for r in rows]
    values = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
    _save_barh(path, labels, values, title=title, xlabel=xlabel, top_k=len(labels))


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


# ============================================================
# Feature naming / grouping
# ============================================================

def infer_feature_group(feature: str) -> str:
    f = feature.lower()

    if "rsi" in f:
        return "rsi"
    if "bollinger" in f:
        return "bollinger"
    if "imb" in f or "imbalance" in f:
        return "imbalance"
    if "dmi" in f:
        return "dmi"
    if "corr" in f:
        return "corr"
    if "obv" in f:
        return "obv"
    if "psy" in f:
        return "psy"
    if re.search(r"(^|_)roc($|_)", f):
        return "roc"
    if re.search(r"(^|_)mtm($|_)", f):
        return "mtm"
    if re.search(r"(^|_)mom($|_)", f) or "amt_mom" in f:
        return "momentum"
    if re.search(r"(^|_)cr($|_)", f):
        return "cr"
    if re.search(r"(^|_)br($|_)", f):
        return "br"
    if re.search(r"(^|_)vr($|_)", f):
        return "vr"
    if "wpr" in f:
        return "wpr"
    if "pos" in f:
        return "position"
    if "b36" in f or "b612" in f:
        return "b_signal"
    if "amt" in f or "vol" in f or "volume" in f:
        return "volume_amount"

    return "other"


def infer_feature_tags(feature: str) -> dict[str, Any]:
    f = feature.lower()
    return {
        "group": infer_feature_group(feature),
        "neutralized": int("neutralized" in f),
        "zscore": int("zscore" in f),
        "raw_name": feature,
    }


# ============================================================
# Projection input decoder
# ============================================================

def decode_projection_columns(model: torch.nn.Module, feature_names: list[str]) -> list[dict[str, Any]]:
    """
    Decode input_projection columns into:
        raw:feature
        span4:feature
        span8:feature
        ...
        cat_or_extra:col
    """
    projection = _get_input_projection(model)
    in_dim = int(projection.weight.shape[1])

    decoded: list[dict[str, Any]] = []

    if _has_ema_bank(model):
        bank = _get_ema_bank(model)
        assert bank is not None

        spans = [int(s) for s in getattr(bank, "spans")]
        num_features = int(getattr(bank, "num_features", len(feature_names)))
        include_raw = bool(getattr(model.encoder, "include_raw_numeric", True))

        col = 0

        if include_raw:
            for j, name in enumerate(feature_names):
                decoded.append({
                    "col": col,
                    "kind": "raw",
                    "span": "",
                    "feature": name,
                    "feature_index": j,
                    **infer_feature_tags(name),
                })
                col += 1

        for span in spans:
            for j, name in enumerate(feature_names):
                decoded.append({
                    "col": col,
                    "kind": "cnn",
                    "span": span,
                    "feature": name,
                    "feature_index": j,
                    **infer_feature_tags(name),
                })
                col += 1

        while col < in_dim:
            decoded.append({
                "col": col,
                "kind": "categorical_or_extra",
                "span": "",
                "feature": f"extra_col_{col}",
                "feature_index": -1,
                "group": "categorical_or_extra",
                "neutralized": 0,
                "zscore": 0,
                "raw_name": f"extra_col_{col}",
            })
            col += 1

    else:
        if len(feature_names) != in_dim:
            raise ValueError(
                f"No EMA bank: expected len(feature_names)==input_projection.in_dim, "
                f"got {len(feature_names)} vs {in_dim}."
            )

        for j, name in enumerate(feature_names):
            decoded.append({
                "col": j,
                "kind": "raw",
                "span": "",
                "feature": name,
                "feature_index": j,
                **infer_feature_tags(name),
            })

    return decoded


# ============================================================
# Effective temporal loading
# ============================================================

def _kernel_to_lag(kernel_oldest_to_current: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Conv kernel order in your module:
        index 0 sees oldest x_{t-span+1}
        index -1 sees current x_t

    Return lag response:
        lag_response[0] = current loading
        lag_response[span-1] = oldest loading
    """
    k = np.asarray(kernel_oldest_to_current, dtype=np.float64)
    out = np.zeros(max_lag + 1, dtype=np.float64)

    span = len(k)
    for p, val in enumerate(k):
        lag = span - 1 - p
        out[lag] += val

    return out


def compute_effective_temporal_loading(
    model: torch.nn.Module,
    feature_names: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Return effective[d, j, lag]:

        hidden_dim d
        feature j
        lag 0..max_span-1

    This combines:
        raw branch weight at lag 0
        CNN span branch weight * learned depthwise kernel
    """
    projection = _get_input_projection(model)
    W = projection.weight.detach().float().cpu().numpy()
    hidden_dim, in_dim = W.shape

    num_features = len(feature_names)

    if not _has_ema_bank(model):
        effective = np.zeros((hidden_dim, num_features, 1), dtype=np.float64)
        effective[:, :, 0] = W[:, :num_features]
        meta = {
            "has_ema_bank": False,
            "spans": [],
            "max_lag": 0,
            "include_raw_numeric": True,
            "input_dim": in_dim,
        }
        return effective, meta

    bank = _get_ema_bank(model)
    assert bank is not None

    spans = [int(s) for s in getattr(bank, "spans")]
    max_lag = max(spans) - 1
    include_raw = bool(getattr(model.encoder, "include_raw_numeric", True))
    bank_num_features = int(getattr(bank, "num_features"))

    if bank_num_features != num_features:
        raise ValueError(
            f"bank.num_features={bank_num_features}, len(feature_names)={num_features}"
        )

    effective = np.zeros((hidden_dim, num_features, max_lag + 1), dtype=np.float64)

    col = 0

    if include_raw:
        effective[:, :, 0] += W[:, col:col + num_features]
        col += num_features

    for span_idx, span in enumerate(spans):
        conv = bank.convs[span_idx]
        kernels = conv.weight.detach().float().cpu().numpy()[:, 0, :]  # [F, span]

        span_W = W[:, col:col + num_features]  # [D, F]

        for j in range(num_features):
            lag_kernel = _kernel_to_lag(kernels[j], max_lag)
            effective[:, j, :] += span_W[:, j:j + 1] * lag_kernel[None, :]

        col += num_features

    meta = {
        "has_ema_bank": True,
        "spans": spans,
        "max_lag": max_lag,
        "include_raw_numeric": include_raw,
        "input_dim": in_dim,
        "used_numeric_temporal_dim": col,
        "categorical_or_extra_dim": max(0, in_dim - col),
    }

    return effective, meta


def _lag_stats_from_response(resp: np.ndarray) -> dict[str, float]:
    """
    resp: [lag]
    """
    r = np.asarray(resp, dtype=np.float64)
    lags = np.arange(len(r), dtype=np.float64)

    abs_sum = max(float(np.abs(r).sum()), 1e-12)
    pos = float(np.clip(r, 0, None).sum())
    neg = float(np.clip(-r, 0, None).sum())
    signed_sum = float(r.sum())
    highpass_score = float(1.0 - min(abs(signed_sum) / abs_sum, 1.0))

    return {
        "signed_sum": signed_sum,
        "abs_sum": abs_sum,
        "pos_mass": pos,
        "neg_mass": neg,
        "neg_mass_ratio": neg / abs_sum,
        "abs_center_lag": float((np.abs(r) * lags).sum() / abs_sum),
        "current_loading": float(r[0]),
        "max_abs_loading": float(np.abs(r).max()),
        "argmax_abs_lag": int(np.abs(r).argmax()),
        "highpass_score": highpass_score,
        "l2": float(np.linalg.norm(r)),
    }


def analyze_effective_temporal_loading(
    *,
    model: torch.nn.Module,
    feature_names: list[str],
    output_dir: Path,
    top_k: int,
    n_clusters: int,
) -> dict[str, Any]:
    figs = output_dir / "figures"
    tables = output_dir / "tables"
    arrays = output_dir / "arrays"
    feature_figs = figs / "effective_feature_lag_response_top"

    for p in [figs, tables, arrays, feature_figs]:
        p.mkdir(parents=True, exist_ok=True)

    effective, meta = compute_effective_temporal_loading(model, feature_names)
    np.save(arrays / "effective_temporal_loading.npy", effective)

    D, F, L = effective.shape
    lags = np.arange(L)

    rows: list[dict[str, Any]] = []

    feature_vectors = effective.transpose(1, 0, 2).reshape(F, D * L)
    feature_total_l2 = np.linalg.norm(feature_vectors, axis=1)

    feature_lag_abs = np.abs(effective).sum(axis=0)  # [F, L]
    feature_lag_signed_sum = effective.sum(axis=0)   # [F, L], signed across hidden dims

    # dominant hidden dim per feature by response norm
    hidden_feature_norm = np.linalg.norm(effective, axis=2)  # [D, F]

    for j, feature in enumerate(feature_names):
        dominant_hidden = int(np.argmax(hidden_feature_norm[:, j]))
        dominant_resp = effective[dominant_hidden, j, :]
        agg_abs_resp = feature_lag_abs[j, :]

        dom_stats = _lag_stats_from_response(dominant_resp)
        abs_stats = _lag_stats_from_response(agg_abs_resp)

        tags = infer_feature_tags(feature)

        rows.append({
            "feature": feature,
            "feature_index": j,
            "group": tags["group"],
            "neutralized": tags["neutralized"],
            "zscore": tags["zscore"],
            "total_effective_l2": float(feature_total_l2[j]),
            "dominant_hidden_dim": dominant_hidden,
            "dominant_hidden_l2": float(hidden_feature_norm[dominant_hidden, j]),
            "dominant_signed_sum": dom_stats["signed_sum"],
            "dominant_current_loading": dom_stats["current_loading"],
            "dominant_abs_center_lag": dom_stats["abs_center_lag"],
            "dominant_neg_mass_ratio": dom_stats["neg_mass_ratio"],
            "dominant_highpass_score": dom_stats["highpass_score"],
            "aggregate_abs_center_lag": abs_stats["abs_center_lag"],
            "aggregate_argmax_abs_lag": abs_stats["argmax_abs_lag"],
            "aggregate_abs_sum": abs_stats["abs_sum"],
        })

    rows = sorted(rows, key=lambda r: r["total_effective_l2"], reverse=True)
    _write_csv(tables / "effective_feature_temporal_summary.csv", rows)

    # Plot top features: aggregate absolute response + signed dominant hidden response.
    top_indices = [int(r["feature_index"]) for r in rows[:top_k]]

    for rank, j in enumerate(top_indices):
        feature = feature_names[j]
        dominant_hidden = int(rows[rank]["dominant_hidden_dim"])

        agg_abs = feature_lag_abs[j, :]
        if agg_abs.max() > 0:
            agg_abs_norm = agg_abs / agg_abs.max()
        else:
            agg_abs_norm = agg_abs

        dom = effective[dominant_hidden, j, :]
        denom = max(float(np.abs(dom).max()), 1e-12)
        dom_norm = dom / denom

        _save_line_plot(
            feature_figs / f"rank{rank:02d}_{_safe_name(feature)}.png",
            lags,
            [
                ("aggregate_abs_response_norm", agg_abs_norm),
                (f"signed_dominant_hidden_{dominant_hidden}_norm", dom_norm),
            ],
            title=f"Effective lag response: {feature}",
            xlabel="lag",
            ylabel="normalized response",
        )

    # Effective feature clustering.
    if F >= 2 and D * L >= 2:
        X = _cosine_normalize_rows(feature_vectors)
        coords, explained = _pca_2d(X)

        annotate = np.argsort(feature_total_l2)[::-1][:min(top_k, F)]

        _save_scatter(
            figs / "effective_feature_cluster_pca.png",
            coords,
            title=f"Effective feature loading PCA; explained={explained[0]:.2%},{explained[1]:.2%}",
            labels=feature_names,
            annotate_indices=annotate,
        )

        # simple kmeans
        labels = _kmeans(X, n_clusters, seed=0)
        cluster_rows = []
        for j, feature in enumerate(feature_names):
            cluster_rows.append({
                "feature": feature,
                "feature_index": j,
                "cluster": int(labels[j]),
                "group": infer_feature_group(feature),
                "pc1": float(coords[j, 0]),
                "pc2": float(coords[j, 1]),
                "total_effective_l2": float(feature_total_l2[j]),
            })

        cluster_rows = sorted(cluster_rows, key=lambda r: (r["cluster"], -r["total_effective_l2"]))
        _write_csv(tables / "effective_feature_clusters.csv", cluster_rows)

        pca_explained = explained.tolist()
    else:
        pca_explained = None

    # Top feature x lag heatmap
    heat = feature_lag_abs[top_indices, :]
    row_labels = [feature_names[j] for j in top_indices]

    _save_heatmap(
        figs / "effective_feature_lag_abs_heatmap_top.png",
        heat,
        title="Top features: aggregate |effective lag response|",
        xlabel="lag",
        ylabel="feature",
        xticklabels=[str(i) for i in lags],
        yticklabels=row_labels,
    )

    return {
        "status": "ok",
        "meta": meta,
        "shape": list(effective.shape),
        "top_features": rows[:20],
        "pca_explained": pca_explained,
    }


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


# ============================================================
# Feature group summary
# ============================================================

def analyze_feature_groups(
    *,
    model: torch.nn.Module,
    feature_names: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    figs = output_dir / "figures"
    tables = output_dir / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    effective, meta = compute_effective_temporal_loading(model, feature_names)
    D, F, L = effective.shape
    lags = np.arange(L)

    feature_groups = [infer_feature_group(f) for f in feature_names]
    groups = sorted(set(feature_groups))

    group_rows = []
    group_lag_heat = []

    for group in groups:
        idx = [i for i, g in enumerate(feature_groups) if g == group]
        if not idx:
            continue

        sub = effective[:, idx, :]  # [D, n, L]
        abs_lag = np.abs(sub).sum(axis=(0, 1))  # [L]
        total_l2 = float(np.linalg.norm(sub))

        abs_sum = max(float(abs_lag.sum()), 1e-12)
        center_lag = float((abs_lag * lags).sum() / abs_sum)
        argmax_lag = int(abs_lag.argmax())

        group_rows.append({
            "group": group,
            "num_features": len(idx),
            "total_effective_l2": total_l2,
            "mean_effective_l2_per_feature": total_l2 / max(len(idx), 1),
            "abs_center_lag": center_lag,
            "argmax_abs_lag": argmax_lag,
            "features": "|".join(feature_names[i] for i in idx),
        })

        group_lag_heat.append(abs_lag / max(float(abs_lag.max()), 1e-12))

    group_rows = sorted(group_rows, key=lambda r: r["total_effective_l2"], reverse=True)
    _write_csv(tables / "feature_group_effective_temporal_summary.csv", group_rows)

    _save_group_bar(
        figs / "feature_group_total_effective_l2.png",
        group_rows,
        "total_effective_l2",
        title="Feature group total effective temporal loading",
        xlabel="||effective loading||2",
    )

    _save_group_bar(
        figs / "feature_group_center_lag.png",
        sorted(group_rows, key=lambda r: r["abs_center_lag"], reverse=True),
        "abs_center_lag",
        title="Feature group effective center lag",
        xlabel="abs-weighted center lag",
    )

    if group_lag_heat:
        group_order = [r["group"] for r in group_rows]
        group_to_heat = {
            row["group"]: group_lag_heat[groups.index(row["group"])]
            for row in group_rows
            if row["group"] in groups
        }
        heat = np.asarray([group_to_heat[g] for g in group_order], dtype=np.float64)

        _save_heatmap(
            figs / "feature_group_lag_response_heatmap.png",
            heat,
            title="Feature group normalized |effective lag response|",
            xlabel="lag",
            ylabel="group",
            xticklabels=[str(i) for i in lags],
            yticklabels=group_order,
        )

    return {
        "status": "ok",
        "groups": group_rows,
        "meta": meta,
    }


# ============================================================
# Hidden channel interpretation
# ============================================================

def analyze_hidden_channels(
    *,
    model: torch.nn.Module,
    feature_names: list[str],
    output_dir: Path,
    top_terms: int,
) -> dict[str, Any]:
    tables = output_dir / "tables"
    figs = output_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    projection = _get_input_projection(model)
    W = projection.weight.detach().float().cpu().numpy()
    bias = None
    if getattr(projection, "bias", None) is not None:
        bias = projection.bias.detach().float().cpu().numpy()

    decoded = decode_projection_columns(model, feature_names)

    rows = []

    for d in range(W.shape[0]):
        row = W[d]
        pos_idx = np.argsort(row)[::-1][:top_terms]
        neg_idx = np.argsort(row)[:top_terms]

        def fmt(cols: np.ndarray) -> tuple[str, str, str, str]:
            terms = []
            groups = []
            spans = []
            vals = []

            for c in cols:
                info = decoded[int(c)]
                kind = info["kind"]
                feature = info["feature"]
                span = info["span"]
                group = info["group"]
                val = row[int(c)]

                if kind == "cnn":
                    term = f"span{span}:{feature}"
                    spans.append(f"span{span}")
                elif kind == "raw":
                    term = f"raw:{feature}"
                    spans.append("raw")
                else:
                    term = f"{kind}:{feature}"
                    spans.append(kind)

                terms.append(term)
                groups.append(group)
                vals.append(f"{val:.6g}")

            return "|".join(terms), "|".join(vals), "|".join(groups), "|".join(spans)

        pos_terms, pos_vals, pos_groups, pos_spans = fmt(pos_idx)
        neg_terms, neg_vals, neg_groups, neg_spans = fmt(neg_idx)

        # dominant group/span by absolute loading
        group_mass: dict[str, float] = defaultdict(float)
        span_mass: dict[str, float] = defaultdict(float)

        for c, val in enumerate(row):
            info = decoded[c]
            group_mass[str(info["group"])] += abs(float(val))

            if info["kind"] == "cnn":
                span_key = f"span{info['span']}"
            else:
                span_key = str(info["kind"])

            span_mass[span_key] += abs(float(val))

        dominant_group = max(group_mass.items(), key=lambda kv: kv[1])[0]
        dominant_span = max(span_mass.items(), key=lambda kv: kv[1])[0]

        rows.append({
            "hidden_dim": d,
            "weight_l2": float(np.linalg.norm(row)),
            "weight_l1": float(np.abs(row).sum()),
            "bias": float(bias[d]) if bias is not None else 0.0,
            "dominant_group": dominant_group,
            "dominant_span_or_branch": dominant_span,
            "top_positive_terms": pos_terms,
            "top_positive_values": pos_vals,
            "top_positive_groups": pos_groups,
            "top_positive_spans": pos_spans,
            "top_negative_terms": neg_terms,
            "top_negative_values": neg_vals,
            "top_negative_groups": neg_groups,
            "top_negative_spans": neg_spans,
        })

    rows = sorted(rows, key=lambda r: r["weight_l2"], reverse=True)
    _write_csv(tables / "hidden_channel_interpretation.csv", rows)

    # heatmap hidden x group mass for top hidden channels
    top_rows = rows[:min(50, len(rows))]
    groups = sorted({g for r in rows for g in [r["dominant_group"]]})
    group_all = sorted({info["group"] for info in decoded})

    hidden_ids = [int(r["hidden_dim"]) for r in top_rows]
    heat = np.zeros((len(hidden_ids), len(group_all)), dtype=np.float64)

    for ii, d in enumerate(hidden_ids):
        row = W[d]
        for c, val in enumerate(row):
            g = decoded[c]["group"]
            jj = group_all.index(g)
            heat[ii, jj] += abs(float(val))

        if heat[ii].max() > 0:
            heat[ii] /= heat[ii].max()

    _save_heatmap(
        figs / "hidden_channel_group_mass_heatmap_top.png",
        heat,
        title="Top hidden channels: normalized group loading mass",
        xlabel="feature group",
        ylabel="hidden dim",
        xticklabels=group_all,
        yticklabels=[str(d) for d in hidden_ids],
    )

    return {
        "status": "ok",
        "top_hidden_channels": rows[:20],
    }


# ============================================================
# Projection contribution decomposition
# ============================================================

def analyze_projection_contribution_by_decoded_terms(
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
    """
    Computes gradient * projection_input contribution.
    Then aggregates by decoded kind/span/group/feature.

    This is closer to "what terms drive prediction" than raw feature grad only.
    """
    tables = output_dir / "tables"
    figs = output_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    projection = _get_input_projection(model)
    decoded = decode_projection_columns(model, feature_names)
    in_dim = int(projection.weight.shape[1])

    col_sum = np.zeros(in_dim, dtype=np.float64)
    col_abs_sum = np.zeros(in_dim, dtype=np.float64)
    n_tokens = 0

    captured: dict[str, torch.Tensor] = {}

    def hook(_module, inputs, output):
        x = inputs[0]
        x.retain_grad()
        captured["projection_input"] = x

    handle = projection.register_forward_hook(hook)

    model.eval()

    try:
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break

            batch = _to_device(batch, device)

            if normalize_features:
                batch = _normalize_batch_features(batch, feature_stats, numeric_groups, device)

            model.zero_grad(set_to_none=True)
            captured.clear()

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

            proj_in = captured["projection_input"]
            grad = proj_in.grad

            if grad is None:
                continue

            contrib = proj_in.detach() * grad.detach()

            flat = _flatten_last_dim(contrib)

            valid_proj = None if ignore_padding_mask else _valid_flat_mask_from_padding(batch, proj_in)
            if valid_proj is not None:
                flat = flat[valid_proj]

            arr = _detach_np(flat)
            col_sum += arr.sum(axis=0)
            col_abs_sum += np.abs(arr).sum(axis=0)
            n_tokens += arr.shape[0]

    finally:
        handle.remove()

    if n_tokens == 0:
        return {"status": "skipped_no_tokens"}

    col_mean = col_sum / n_tokens
    col_mean_abs = col_abs_sum / n_tokens

    col_rows = []
    feature_mass: dict[str, dict[str, Any]] = {}
    group_mass: dict[str, dict[str, Any]] = {}
    span_mass: dict[str, dict[str, Any]] = {}

    def acc(store: dict[str, dict[str, Any]], key: str, signed: float, absval: float):
        if key not in store:
            store[key] = {"key": key, "mean_signed": 0.0, "mean_abs": 0.0}
        store[key]["mean_signed"] += signed
        store[key]["mean_abs"] += absval

    for c, info in enumerate(decoded):
        kind = info["kind"]
        feature = info["feature"]
        group = info["group"]

        if kind == "cnn":
            span_key = f"span{info['span']}"
            term = f"{span_key}:{feature}"
        else:
            span_key = kind
            term = f"{kind}:{feature}"

        signed = float(col_mean[c])
        absval = float(col_mean_abs[c])

        col_rows.append({
            "projection_input_col": c,
            "term": term,
            "kind": kind,
            "span_or_branch": span_key,
            "feature": feature,
            "feature_index": info["feature_index"],
            "group": group,
            "mean_signed_grad_x_input": signed,
            "mean_abs_grad_x_input": absval,
        })

        acc(feature_mass, feature, signed, absval)
        acc(group_mass, group, signed, absval)
        acc(span_mass, span_key, signed, absval)

    col_rows = sorted(col_rows, key=lambda r: r["mean_abs_grad_x_input"], reverse=True)
    _write_csv(tables / "projection_input_grad_x_contribution.csv", col_rows)

    feature_rows = []
    for k, v in feature_mass.items():
        feature_rows.append({
            "feature": k,
            "group": infer_feature_group(k),
            "mean_signed_grad_x_input": v["mean_signed"],
            "mean_abs_grad_x_input": v["mean_abs"],
        })
    feature_rows = sorted(feature_rows, key=lambda r: r["mean_abs_grad_x_input"], reverse=True)
    _write_csv(tables / "feature_total_grad_x_contribution.csv", feature_rows)

    group_rows = []
    for k, v in group_mass.items():
        group_rows.append({
            "group": k,
            "mean_signed_grad_x_input": v["mean_signed"],
            "mean_abs_grad_x_input": v["mean_abs"],
        })
    group_rows = sorted(group_rows, key=lambda r: r["mean_abs_grad_x_input"], reverse=True)
    _write_csv(tables / "group_grad_x_contribution.csv", group_rows)

    span_rows = []
    for k, v in span_mass.items():
        span_rows.append({
            "span_or_branch": k,
            "mean_signed_grad_x_input": v["mean_signed"],
            "mean_abs_grad_x_input": v["mean_abs"],
        })
    span_rows = sorted(span_rows, key=lambda r: r["mean_abs_grad_x_input"], reverse=True)
    _write_csv(tables / "span_branch_grad_x_contribution.csv", span_rows)

    _save_barh(
        figs / "feature_total_grad_x_contribution_top.png",
        [r["feature"] for r in feature_rows],
        np.asarray([r["mean_abs_grad_x_input"] for r in feature_rows]),
        title="Feature total contribution: projection-input grad x input",
        xlabel="mean abs contribution",
        top_k=min(50, len(feature_rows)),
    )

    _save_barh(
        figs / "group_grad_x_contribution.png",
        [r["group"] for r in group_rows],
        np.asarray([r["mean_abs_grad_x_input"] for r in group_rows]),
        title="Group contribution: projection-input grad x input",
        xlabel="mean abs contribution",
        top_k=len(group_rows),
    )

    _save_barh(
        figs / "span_branch_grad_x_contribution.png",
        [r["span_or_branch"] for r in span_rows],
        np.asarray([r["mean_abs_grad_x_input"] for r in span_rows]),
        title="Span / branch contribution: projection-input grad x input",
        xlabel="mean abs contribution",
        top_k=len(span_rows),
    )

    return {
        "status": "ok",
        "n_tokens": int(n_tokens),
        "top_terms": col_rows[:20],
        "top_features": feature_rows[:20],
        "groups": group_rows,
        "spans": span_rows,
    }


# ============================================================
# PDP / perturbation curves
# ============================================================

def _resolve_feature_local_index(feature_names: list[str], feature: str) -> int:
    if feature not in feature_names:
        raise KeyError(f"Feature {feature!r} not in feature_names.")
    return feature_names.index(feature)


def _set_feature_value_in_batch(
    batch: dict[str, Any],
    numeric_groups: list[str],
    feature_global_idx: int,
    value: float,
) -> dict[str, Any]:
    offset = 0

    for group in numeric_groups:
        x = batch["features"].get(group)

        if x is None:
            continue

        width = x.shape[-1]

        if offset <= feature_global_idx < offset + width:
            local = feature_global_idx - offset
            x = x.clone()
            x[..., local] = float(value)
            batch["features"][group] = x
            return batch

        offset += width

    raise IndexError(f"feature_global_idx={feature_global_idx} not found in numeric groups.")


def _predict_flat(
    *,
    model: torch.nn.Module,
    batch: dict[str, Any],
    target_key: str,
    pred_key: str | None,
    ignore_padding_mask: bool,
) -> torch.Tensor:
    out = model(batch)
    pred = _extract_pred(out, pred_key, target_key).float()

    valid = None if ignore_padding_mask else _valid_flat_mask_from_padding(batch, pred)

    flat = pred.reshape(-1)

    if valid is not None and pred.ndim >= 3 and pred.shape[-1] == 1:
        flat = flat[valid]
    elif valid is not None and pred.ndim == 2:
        flat = flat[valid]

    return flat


def run_pdp_curves(
    *,
    model: torch.nn.Module,
    loader_builder: Any,
    train_df: pl.DataFrame,
    feature_stats: Mapping[str, Mapping[str, list[float]]],
    numeric_groups: list[str],
    feature_names: list[str],
    output_dir: Path,
    device: torch.device,
    normalize_features: bool,
    target_key: str,
    pred_key: str | None,
    ignore_padding_mask: bool,
    features: list[str],
    max_batches: int,
    num_grid: int,
) -> dict[str, Any]:
    figs = output_dir / "figures" / "pdp_top_features"
    tables = output_dir / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    rows = []

    # Use normalized grid if model inputs are normalized.
    group = numeric_groups[0]
    stats = feature_stats.get(group, {"mean": [0.0] * len(feature_names), "std": [1.0] * len(feature_names)})
    means = np.asarray(stats["mean"], dtype=np.float64)
    stds = np.maximum(np.asarray(stats["std"], dtype=np.float64), 1e-12)

    model.eval()

    for feature in features:
        if feature not in feature_names:
            continue

        j = _resolve_feature_local_index(feature_names, feature)

        raw_vals = train_df.select(pl.col(feature)).to_numpy().reshape(-1)
        raw_vals = raw_vals[np.isfinite(raw_vals)]

        if raw_vals.size == 0:
            continue

        qs = np.linspace(0.01, 0.99, num_grid)
        raw_grid = np.quantile(raw_vals, qs)

        if normalize_features:
            model_grid = (raw_grid - means[j]) / stds[j]
        else:
            model_grid = raw_grid

        baseline_chunks = []
        grid_sum = np.zeros(num_grid, dtype=np.float64)
        grid_count = np.zeros(num_grid, dtype=np.float64)

        # First baseline over same batches.
        loader = loader_builder()

        with torch.no_grad():
            saved_batches = []

            for batch_idx, batch in enumerate(loader):
                if batch_idx >= max_batches:
                    break

                batch = _to_device(batch, device)

                if normalize_features:
                    batch = _normalize_batch_features(batch, feature_stats, numeric_groups, device)

                base_pred = _predict_flat(
                    model=model,
                    batch=batch,
                    target_key=target_key,
                    pred_key=pred_key,
                    ignore_padding_mask=ignore_padding_mask,
                )

                baseline_chunks.append(_detach_np(base_pred))
                saved_batches.append(batch)

            if not saved_batches:
                continue

            base_mean = float(np.concatenate(baseline_chunks).mean())

            for gi, val in enumerate(model_grid):
                pred_vals = []

                for batch in saved_batches:
                    b2 = {
                        k: v for k, v in batch.items()
                    }
                    b2["features"] = {
                        k: v for k, v in batch["features"].items()
                    }

                    b2 = _set_feature_value_in_batch(
                        b2,
                        numeric_groups,
                        feature_global_idx=j,
                        value=float(val),
                    )

                    pred = _predict_flat(
                        model=model,
                        batch=b2,
                        target_key=target_key,
                        pred_key=pred_key,
                        ignore_padding_mask=ignore_padding_mask,
                    )

                    pred_vals.append(_detach_np(pred))

                pred_all = np.concatenate(pred_vals)
                grid_sum[gi] = float(pred_all.mean())
                grid_count[gi] = float(pred_all.size)

                rows.append({
                    "feature": feature,
                    "feature_index": j,
                    "group": infer_feature_group(feature),
                    "quantile": float(qs[gi]),
                    "raw_value": float(raw_grid[gi]),
                    "model_input_value": float(val),
                    "mean_pred": float(grid_sum[gi]),
                    "mean_pred_delta_vs_baseline": float(grid_sum[gi] - base_mean),
                    "baseline_mean_pred": base_mean,
                    "n": int(grid_count[gi]),
                })

        y = np.asarray([r["mean_pred_delta_vs_baseline"] for r in rows if r["feature"] == feature], dtype=np.float64)
        x = raw_grid

        plt.figure(figsize=(7, 4.5))
        plt.plot(x, y, marker="o")
        plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.xlabel(f"{feature} raw value")
        plt.ylabel("mean pred delta")
        plt.title(f"PDP / perturbation curve: {feature}")
        plt.tight_layout()
        plt.savefig(figs / f"{_safe_name(feature)}.png", dpi=180)
        plt.close()

    _write_csv(tables / "pdp_feature_curves.csv", rows)

    return {
        "status": "ok",
        "features": features,
        "num_rows": len(rows),
    }


# ============================================================
# Group ablation
# ============================================================

def run_group_ablation(
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
    ignore_padding_mask: bool,
) -> dict[str, Any]:
    tables = output_dir / "tables"
    figs = output_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    group_to_indices: dict[str, list[int]] = defaultdict(list)
    for j, f in enumerate(feature_names):
        group_to_indices[infer_feature_group(f)].append(j)

    groups = sorted(group_to_indices.keys())

    def eval_with_zero_indices(indices: list[int] | None) -> dict[str, float]:
        loader = loader_builder()
        pred_chunks = []
        target_chunks = []
        weight_chunks = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= max_batches:
                    break

                batch = _to_device(batch, device)

                if normalize_features:
                    batch = _normalize_batch_features(batch, feature_stats, numeric_groups, device)

                if indices:
                    for idx in indices:
                        batch = _set_feature_value_in_batch(
                            batch,
                            numeric_groups,
                            feature_global_idx=idx,
                            value=0.0,
                        )

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

                pred_chunks.append(_detach_np(pred_flat))
                target_chunks.append(_detach_np(target_flat))
                weight_chunks.append(_detach_np(weight_flat))

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

    baseline = eval_with_zero_indices(None)

    rows = []
    for group in groups:
        metrics = eval_with_zero_indices(group_to_indices[group])

        rows.append({
            "group": group,
            "num_features": len(group_to_indices[group]),
            "baseline_weighted_cosine": baseline.get("weighted_cosine", float("nan")),
            "ablated_weighted_cosine": metrics.get("weighted_cosine", float("nan")),
            "delta_weighted_cosine": baseline.get("weighted_cosine", float("nan")) - metrics.get("weighted_cosine", float("nan")),
            "baseline_r2": baseline.get("weighted_r2_zero_benchmark", float("nan")),
            "ablated_r2": metrics.get("weighted_r2_zero_benchmark", float("nan")),
            "delta_r2": baseline.get("weighted_r2_zero_benchmark", float("nan")) - metrics.get("weighted_r2_zero_benchmark", float("nan")),
            "baseline_mse": baseline.get("mse", float("nan")),
            "ablated_mse": metrics.get("mse", float("nan")),
            "delta_mse": metrics.get("mse", float("nan")) - baseline.get("mse", float("nan")),
            "features": "|".join(feature_names[i] for i in group_to_indices[group]),
        })

    rows = sorted(rows, key=lambda r: abs(r["delta_weighted_cosine"]), reverse=True)
    _write_csv(tables / "group_ablation.csv", rows)

    _save_barh(
        figs / "group_ablation_delta_cosine.png",
        [r["group"] for r in rows],
        np.asarray([r["delta_weighted_cosine"] for r in rows], dtype=np.float64),
        title="Group ablation: baseline cosine - ablated cosine",
        xlabel="delta weighted cosine",
        top_k=len(rows),
    )

    return {
        "status": "ok",
        "baseline": baseline,
        "rows": rows,
    }


# ============================================================
# Lag occlusion
# ============================================================

def run_lag_occlusion(
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
    pred_key: str | None,
    ignore_padding_mask: bool,
    lags: list[int],
) -> dict[str, Any]:
    """
    Coarse interventional lag sensitivity:
        for lag L, zero x_{t-L} and measure |pred_t - pred_t_perturbed|.

    This is not a perfect causal attribution because zeroing an input position can
    affect multiple future outputs, but it is very useful as a sanity check.
    """
    tables = output_dir / "tables"
    figs = output_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    rows = []

    for lag in lags:
        loader = loader_builder()

        diffs = []
        signed_diffs = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= max_batches:
                    break

                batch = _to_device(batch, device)

                if normalize_features:
                    batch = _normalize_batch_features(batch, feature_stats, numeric_groups, device)

                pred0 = _extract_pred(model(batch), pred_key, target_key).float()

                if pred0.ndim == 2:
                    pred0 = pred0.unsqueeze(-1)

                B, T = pred0.shape[0], pred0.shape[1]

                if lag >= T:
                    continue

                b2 = {k: v for k, v in batch.items()}
                b2["features"] = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch["features"].items()}

                for group in numeric_groups:
                    x = b2["features"].get(group)
                    if x is None:
                        continue
                    x = x.clone()

                    # Zero positions s=0..T-lag-1; evaluate outputs t=s+lag.
                    x[:, :T - lag, :] = 0.0
                    b2["features"][group] = x

                pred1 = _extract_pred(model(b2), pred_key, target_key).float()
                if pred1.ndim == 2:
                    pred1 = pred1.unsqueeze(-1)

                p0 = pred0[:, lag:, :]
                p1 = pred1[:, lag:, :]

                if not ignore_padding_mask and "padding_mask" in batch:
                    valid = batch["padding_mask"][:, lag:].bool()
                    delta = (p1 - p0).squeeze(-1)[valid]
                else:
                    delta = (p1 - p0).reshape(-1)

                if delta.numel() == 0:
                    continue

                diffs.append(_detach_np(delta.abs()))
                signed_diffs.append(_detach_np(delta))

        if diffs:
            all_abs = np.concatenate(diffs)
            all_signed = np.concatenate(signed_diffs)

            rows.append({
                "lag": lag,
                "mean_abs_pred_delta": float(all_abs.mean()),
                "mean_signed_pred_delta": float(all_signed.mean()),
                "p90_abs_pred_delta": float(np.quantile(all_abs, 0.90)),
                "n": int(all_abs.size),
            })

    _write_csv(tables / "lag_occlusion.csv", rows)

    if rows:
        xs = np.asarray([r["lag"] for r in rows], dtype=np.float64)
        ys = np.asarray([r["mean_abs_pred_delta"] for r in rows], dtype=np.float64)

        plt.figure(figsize=(7, 4.5))
        plt.plot(xs, ys, marker="o")
        plt.xlabel("lag")
        plt.ylabel("mean |pred delta|")
        plt.title("Whole-model lag occlusion sensitivity")
        plt.tight_layout()
        plt.savefig(figs / "lag_occlusion_mean_abs_delta.png", dpi=180)
        plt.close()

    return {
        "status": "ok",
        "rows": rows,
    }


# ============================================================
# Hidden top activating examples
# ============================================================

def _flatten_meta(batch: dict[str, Any]) -> list[dict[str, Any]]:
    meta = batch.get("meta")
    if not isinstance(meta, list):
        return []

    out = []
    for sample in meta:
        # sample contains arrays for Time and Symbol.
        keys = list(sample.keys())
        time_key = next((k for k in keys if "time" in str(k).lower()), None)
        symbol_key = next((k for k in keys if "symbol" in str(k).lower()), None)

        times = sample.get(time_key, []) if time_key is not None else []
        symbols = sample.get(symbol_key, []) if symbol_key is not None else []

        n = min(len(times), len(symbols)) if len(times) and len(symbols) else max(len(times), len(symbols))
        for i in range(n):
            out.append({
                "time": str(times[i]) if len(times) else "",
                "symbol": str(symbols[i]) if len(symbols) else "",
            })

    return out


def collect_hidden_top_examples(
    *,
    model: torch.nn.Module,
    loader: Any,
    feature_stats: Mapping[str, Mapping[str, list[float]]],
    numeric_groups: list[str],
    output_dir: Path,
    device: torch.device,
    max_batches: int,
    normalize_features: bool,
    target_key: str,
    target_index: int,
    pred_key: str | None,
    ignore_padding_mask: bool,
    top_hidden_dims: int,
    examples_per_hidden: int,
) -> dict[str, Any]:
    tables = output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    projection = _get_input_projection(model)
    W = projection.weight.detach().float().cpu().numpy()
    hidden_norm = np.linalg.norm(W, axis=1)
    hidden_dims = np.argsort(hidden_norm)[::-1][:top_hidden_dims]

    heaps: dict[int, list[tuple[float, dict[str, Any]]]] = {int(d): [] for d in hidden_dims}

    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        captured["encoder_output"] = output.detach()

    handle = projection.register_forward_hook(hook)

    model.eval()

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
                pred = _extract_pred(out, pred_key, target_key).float()
                target = _extract_target(batch, target_key, target_index)

                if "encoder_output" not in captured:
                    continue

                z = captured["encoder_output"]  # [B,T,D]
                flat_z = z.reshape(-1, z.shape[-1])

                pred_flat = pred.reshape(-1)
                if target is not None:
                    target_flat = target.reshape(-1)
                else:
                    target_flat = torch.full_like(pred_flat, float("nan"))

                valid = None if ignore_padding_mask else _valid_flat_mask_from_padding(batch, z)

                if valid is not None:
                    flat_z_valid = flat_z[valid]
                    pred_valid = pred_flat[valid]
                    target_valid = target_flat[valid]
                    meta_flat = _flatten_meta(batch)
                    meta_valid = [m for m, keep in zip(meta_flat, valid.detach().cpu().numpy().tolist()) if keep]
                else:
                    flat_z_valid = flat_z
                    pred_valid = pred_flat
                    target_valid = target_flat
                    meta_valid = _flatten_meta(batch)

                z_np = _detach_np(flat_z_valid)
                pred_np = _detach_np(pred_valid)
                target_np = _detach_np(target_valid)

                for d in hidden_dims:
                    d = int(d)
                    vals = z_np[:, d]

                    top_idx = np.argsort(vals)[::-1][:examples_per_hidden]

                    for idx in top_idx:
                        meta_row = meta_valid[idx] if idx < len(meta_valid) else {"time": "", "symbol": ""}
                        item = {
                            "hidden_dim": d,
                            "activation": float(vals[idx]),
                            "pred": float(pred_np[idx]) if idx < len(pred_np) else float("nan"),
                            "target": float(target_np[idx]) if idx < len(target_np) else float("nan"),
                            "time": meta_row.get("time", ""),
                            "symbol": meta_row.get("symbol", ""),
                            "batch_idx": batch_idx,
                        }

                        heap = heaps[d]
                        score = float(vals[idx])
                        if len(heap) < examples_per_hidden:
                            heapq.heappush(heap, (score, item))
                        else:
                            if score > heap[0][0]:
                                heapq.heapreplace(heap, (score, item))

    finally:
        handle.remove()

    rows = []
    for d, heap in heaps.items():
        items = [x[1] for x in sorted(heap, key=lambda p: p[0], reverse=True)]
        for rank, item in enumerate(items):
            item["rank"] = rank
            rows.append(item)

    _write_csv(tables / "hidden_top_activation_examples.csv", rows)

    return {
        "status": "ok",
        "hidden_dims": [int(d) for d in hidden_dims],
        "num_rows": len(rows),
    }


# ============================================================
# CLI / main
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep interpretability analysis for TabularSequenceEncoder.")

    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--split", choices=["train", "val", "auto"], default="val")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--grad-max-batches", type=int, default=50)
    parser.add_argument("--ablation-max-batches", type=int, default=50)
    parser.add_argument("--pdp-max-batches", type=int, default=20)
    parser.add_argument("--lag-max-batches", type=int, default=20)

    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--n-clusters", type=int, default=12)

    parser.add_argument("--no-normalize-features", action="store_true")
    parser.add_argument("--target-normalization", choices=["train_stats", "none"], default="train_stats")
    parser.add_argument("--ignore-padding-mask", action="store_true")
    parser.add_argument("--apply-config-vocab", action="store_true")
    parser.add_argument("--prefer-raw-model", action="store_true")

    parser.add_argument("--run-pdp", action="store_true")
    parser.add_argument("--pdp-top-n", type=int, default=10)
    parser.add_argument("--pdp-grid", type=int, default=11)

    parser.add_argument("--run-group-ablation", action="store_true")

    parser.add_argument("--run-lag-occlusion", action="store_true")
    parser.add_argument("--lags", default="0,1,2,4,8,16,32")

    parser.add_argument("--run-hidden-examples", action="store_true")
    parser.add_argument("--top-hidden-dims", type=int, default=20)
    parser.add_argument("--examples-per-hidden", type=int, default=20)

    parser.add_argument("--skip-contribution", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    output_dir = Path(args.output_dir)
    deep_dir = output_dir / "deep"

    for p in [deep_dir, deep_dir / "figures", deep_dir / "tables", deep_dir / "arrays"]:
        p.mkdir(parents=True, exist_ok=True)

    cfg = load_config_with_inheritance(args.config)
    data_cfg = dict(cfg["data"])

    train_dataset_cfg, val_dataset_cfg = _resolve_dataset_cfgs(data_cfg)
    model_cfg = _inject_vocab_sizes(dict(cfg["model"]), train_dataset_cfg)

    numeric_groups = [str(v) for v in model_cfg.get("numeric_feature_groups", ["continuous"])]

    print("[1/10] Loading train frame...")
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
            print("[2/10] Loading val frame...")
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
            print("[2/10] Loading val frame...")
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

    print("[3/10] Building loader...")
    loader = make_loader()

    print("[4/10] Building model and initializing LazyLinear...")
    device = torch.device(args.device)
    model = build_model(model_cfg)
    model.to(device)

    first_batch = next(iter(loader))
    first_batch = _to_device(first_batch, device)

    if not args.no_normalize_features:
        first_batch = _normalize_batch_features(first_batch, feature_stats, numeric_groups, device)

    with torch.no_grad():
        _ = model(first_batch)

    print("[5/10] Loading checkpoint...")
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
        "deep_dir": str(deep_dir),
        "split_used": split_used,
        "device": str(device),
        "numeric_groups": numeric_groups,
        "num_features": len(feature_names),
        "target_key": target_key,
        "target_index": target_index,
        "pred_key": pred_key,
        "normalize_features": not args.no_normalize_features,
        "target_normalization": args.target_normalization,
        "has_causal_ema_bank": _has_ema_bank(model),
        "checkpoint_info": ckpt_info,
    }

    (deep_dir / "deep_analysis_config.json").write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[6/10] Effective temporal loading...")
    effective_summary = analyze_effective_temporal_loading(
        model=model,
        feature_names=feature_names,
        output_dir=deep_dir,
        top_k=args.top_k,
        n_clusters=args.n_clusters,
    )

    print("[7/10] Feature group summary...")
    group_summary = analyze_feature_groups(
        model=model,
        feature_names=feature_names,
        output_dir=deep_dir,
    )

    print("[8/10] Hidden channel interpretation...")
    hidden_summary = analyze_hidden_channels(
        model=model,
        feature_names=feature_names,
        output_dir=deep_dir,
        top_terms=15,
    )

    contribution_summary = {"status": "skipped"}
    if not args.skip_contribution:
        print("[9/10] Projection-input grad x input decomposition...")
        loader = make_loader()
        contribution_summary = analyze_projection_contribution_by_decoded_terms(
            model=model,
            loader=loader,
            feature_stats=feature_stats,
            numeric_groups=numeric_groups,
            feature_names=feature_names,
            output_dir=deep_dir,
            device=device,
            max_batches=args.grad_max_batches,
            normalize_features=not args.no_normalize_features,
            target_key=target_key,
            pred_key=pred_key,
            ignore_padding_mask=args.ignore_padding_mask,
        )
    else:
        print("[9/10] Contribution decomposition skipped.")

    pdp_summary = {"status": "skipped"}
    if args.run_pdp:
        print("[extra] PDP / perturbation curves...")

        # Use top effective loading features.
        top_features = [row["feature"] for row in effective_summary["top_features"][:args.pdp_top_n]]

        pdp_summary = run_pdp_curves(
            model=model,
            loader_builder=make_loader,
            train_df=train_df,
            feature_stats=feature_stats,
            numeric_groups=numeric_groups,
            feature_names=feature_names,
            output_dir=deep_dir,
            device=device,
            normalize_features=not args.no_normalize_features,
            target_key=target_key,
            pred_key=pred_key,
            ignore_padding_mask=args.ignore_padding_mask,
            features=top_features,
            max_batches=args.pdp_max_batches,
            num_grid=args.pdp_grid,
        )

    group_ablation_summary = {"status": "skipped"}
    if args.run_group_ablation:
        print("[extra] Group ablation...")
        group_ablation_summary = run_group_ablation(
            model=model,
            loader_builder=make_loader,
            feature_stats=feature_stats,
            numeric_groups=numeric_groups,
            feature_names=feature_names,
            output_dir=deep_dir,
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

    lag_occlusion_summary = {"status": "skipped"}
    if args.run_lag_occlusion:
        print("[extra] Lag occlusion...")
        lags = [int(x) for x in str(args.lags).split(",") if x.strip()]

        lag_occlusion_summary = run_lag_occlusion(
            model=model,
            loader_builder=make_loader,
            feature_stats=feature_stats,
            numeric_groups=numeric_groups,
            output_dir=deep_dir,
            device=device,
            max_batches=args.lag_max_batches,
            normalize_features=not args.no_normalize_features,
            target_key=target_key,
            pred_key=pred_key,
            ignore_padding_mask=args.ignore_padding_mask,
            lags=lags,
        )

    hidden_examples_summary = {"status": "skipped"}
    if args.run_hidden_examples:
        print("[extra] Hidden top activation examples...")
        loader = make_loader()

        hidden_examples_summary = collect_hidden_top_examples(
            model=model,
            loader=loader,
            feature_stats=feature_stats,
            numeric_groups=numeric_groups,
            output_dir=deep_dir,
            device=device,
            max_batches=args.max_batches,
            normalize_features=not args.no_normalize_features,
            target_key=target_key,
            target_index=target_index,
            pred_key=pred_key,
            ignore_padding_mask=args.ignore_padding_mask,
            top_hidden_dims=args.top_hidden_dims,
            examples_per_hidden=args.examples_per_hidden,
        )

    summary = {
        "resolved": resolved,
        "effective_temporal_loading": effective_summary,
        "feature_groups": group_summary,
        "hidden_channels": hidden_summary,
        "projection_contribution": contribution_summary,
        "pdp": pdp_summary,
        "group_ablation": group_ablation_summary,
        "lag_occlusion": lag_occlusion_summary,
        "hidden_top_examples": hidden_examples_summary,
    }

    (deep_dir / "deep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md: list[str] = []
    md.append("# Deep Encoder Interpretability Report\n\n")

    md.append("## What this script adds\n\n")
    md.append("- Effective temporal loading: combines raw branch, CNN/EMA bank, and input_projection into one lag response per feature.\n")
    md.append("- Feature group summary: aggregates feature behavior by semantic groups parsed from feature names.\n")
    md.append("- Hidden channel interpretation: decodes each encoder hidden channel into raw/span feature terms.\n")
    md.append("- Projection-input grad × input decomposition: estimates which raw/span/group terms locally drive predictions.\n")
    md.append("- Optional PDP, group ablation, lag occlusion, and hidden top activation examples.\n\n")

    md.append("## Key output files\n\n")
    md.append("### Effective temporal loading\n\n")
    md.append("- `tables/effective_feature_temporal_summary.csv`\n")
    md.append("- `tables/effective_feature_clusters.csv`\n")
    md.append("- `figures/effective_feature_lag_abs_heatmap_top.png`\n")
    md.append("- `figures/effective_feature_cluster_pca.png`\n")
    md.append("- `figures/effective_feature_lag_response_top/*.png`\n\n")

    md.append("### Feature groups\n\n")
    md.append("- `tables/feature_group_effective_temporal_summary.csv`\n")
    md.append("- `figures/feature_group_total_effective_l2.png`\n")
    md.append("- `figures/feature_group_center_lag.png`\n")
    md.append("- `figures/feature_group_lag_response_heatmap.png`\n\n")

    md.append("### Hidden channels\n\n")
    md.append("- `tables/hidden_channel_interpretation.csv`\n")
    md.append("- `figures/hidden_channel_group_mass_heatmap_top.png`\n\n")

    md.append("### Contribution decomposition\n\n")
    md.append("- `tables/projection_input_grad_x_contribution.csv`\n")
    md.append("- `tables/feature_total_grad_x_contribution.csv`\n")
    md.append("- `tables/group_grad_x_contribution.csv`\n")
    md.append("- `tables/span_branch_grad_x_contribution.csv`\n")
    md.append("- `figures/feature_total_grad_x_contribution_top.png`\n")
    md.append("- `figures/group_grad_x_contribution.png`\n")
    md.append("- `figures/span_branch_grad_x_contribution.png`\n\n")

    if args.run_pdp:
        md.append("### PDP\n\n")
        md.append("- `tables/pdp_feature_curves.csv`\n")
        md.append("- `figures/pdp_top_features/*.png`\n\n")

    if args.run_group_ablation:
        md.append("### Group ablation\n\n")
        md.append("- `tables/group_ablation.csv`\n")
        md.append("- `figures/group_ablation_delta_cosine.png`\n\n")

    if args.run_lag_occlusion:
        md.append("### Lag occlusion\n\n")
        md.append("- `tables/lag_occlusion.csv`\n")
        md.append("- `figures/lag_occlusion_mean_abs_delta.png`\n\n")

    if args.run_hidden_examples:
        md.append("### Hidden activation examples\n\n")
        md.append("- `tables/hidden_top_activation_examples.csv`\n\n")

    md.append("## Top effective temporal features\n\n")
    for row in effective_summary["top_features"][:10]:
        md.append(
            f"- `{row['feature']}`: "
            f"group={row['group']}, "
            f"total_l2={row['total_effective_l2']:.6g}, "
            f"center_lag={row['aggregate_abs_center_lag']:.4g}, "
            f"dominant_hidden={row['dominant_hidden_dim']}\n"
        )

    md.append("\n## Top feature groups by effective loading\n\n")
    for row in group_summary["groups"][:10]:
        md.append(
            f"- `{row['group']}`: "
            f"num_features={row['num_features']}, "
            f"total_l2={row['total_effective_l2']:.6g}, "
            f"center_lag={row['abs_center_lag']:.4g}\n"
        )

    if contribution_summary.get("groups"):
        md.append("\n## Top feature groups by local contribution\n\n")
        for row in contribution_summary["groups"][:10]:
            md.append(
                f"- `{row['group']}`: "
                f"mean_abs_grad_x_input={row['mean_abs_grad_x_input']:.6g}, "
                f"mean_signed={row['mean_signed_grad_x_input']:.6g}\n"
            )

    if group_ablation_summary.get("rows"):
        md.append("\n## Group ablation\n\n")
        for row in group_ablation_summary["rows"]:
            md.append(
                f"- `{row['group']}`: "
                f"delta_cos={row['delta_weighted_cosine']:.6g}, "
                f"delta_r2={row['delta_r2']:.6g}, "
                f"num_features={row['num_features']}\n"
            )

    (deep_dir / "deep_summary.md").write_text("".join(md), encoding="utf-8")

    print(f"[done] deep analysis dumped to: {deep_dir}")


if __name__ == "__main__":
    main()