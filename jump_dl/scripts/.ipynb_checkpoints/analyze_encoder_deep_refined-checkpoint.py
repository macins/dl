from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jump_dl.scripts.analyze_encoder_deep as base


# ============================================================
# Refined feature taxonomy
# ============================================================

def infer_feature_group_refined(feature: str) -> str:
    """
    Finer feature grouping.

    Important:
    Ordering matters. For example, amt_mom should be amount_momentum,
    not price_momentum.
    """
    f = feature.lower()

    # Volume / flow first, because names may contain "mom".
    if "imb_vol" in f or "imbalance" in f:
        return "imbalance"
    if "amt_mom" in f:
        return "amount_momentum"
    if "obv" in f:
        return "obv"
    if re.search(r"(^|_)vr($|_)", f):
        return "volume_ratio"

    # Oscillators / bounded-style technical indicators.
    if "rsi" in f:
        return "rsi"
    if "psy" in f:
        return "psy"
    if "wpr" in f:
        return "wpr"
    if "bollinger" in f:
        return "bollinger"
    if "brar" in f or re.search(r"(^|_)br($|_)", f):
        return "brar"
    if re.search(r"(^|_)cr($|_)", f):
        return "cr"

    # Trend direction.
    if "dmi_plus" in f or "dmi_minus" in f or re.search(r"(^|_)dmi($|_)", f):
        return "dmi"

    # Price momentum family, split into subfamilies.
    if re.search(r"(^|_)mom($|_)", f):
        return "price_momentum"
    if re.search(r"(^|_)mtm($|_)", f):
        return "mtm"
    if re.search(r"(^|_)roc($|_)", f):
        return "roc"

    # Cross-sectional correlation / relation.
    if "corr_cs" in f or "corr_" in f:
        return "cross_corr"

    # Position / path shape.
    if "pos" in f:
        return "position"
    if "min_path" in f or "path" in f:
        return "path_shape"

    # Custom band-like signals.
    if "b36" in f or "b612" in f:
        return "custom_band"

    return "other"


def infer_feature_tags_refined(feature: str) -> dict[str, Any]:
    f = feature.lower()
    return {
        "group": infer_feature_group_refined(feature),
        "neutralized": int("neutralized" in f),
        "zscore": int("zscore" in f),
        "raw_name": feature,
    }


# ============================================================
# Helpers
# ============================================================

def _entropy_from_mass(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    total = float(values.sum())
    if total <= 0:
        return 0.0
    p = values / total
    return float(-(p * np.log(p + 1e-12)).sum())


def _safe_ratio(a: float, b: float) -> float:
    return float(a / max(b, 1e-12))


def _group_counts_from_features(feature_names: list[str]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for f in feature_names:
        out[infer_feature_group_refined(f)] += 1
    return dict(out)


def _save_barh(path: Path, labels: list[str], values: np.ndarray, title: str, xlabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)[::-1]

    labels = [labels[i] for i in order][::-1]
    values = values[order][::-1]

    plt.figure(figsize=(10, max(4, 0.32 * len(labels))))
    plt.barh(np.arange(len(labels)), values)
    plt.yticks(np.arange(len(labels)), labels, fontsize=8)
    plt.xlabel(xlabel)
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
    xticklabels: list[str],
    yticklabels: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    mat = np.asarray(mat, dtype=np.float64)

    plt.figure(figsize=(max(8, 0.35 * mat.shape[1]), max(6, 0.18 * mat.shape[0])))
    plt.imshow(mat, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(np.arange(len(xticklabels)), xticklabels, rotation=90, fontsize=7)
    plt.yticks(np.arange(len(yticklabels)), yticklabels, fontsize=6)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


# ============================================================
# Refined feature group summary
# ============================================================

def analyze_feature_groups_refined(
    *,
    model,
    feature_names: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    figs = output_dir / "figures"
    tables = output_dir / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    effective, meta = base.compute_effective_temporal_loading(model, feature_names)
    D, F, L = effective.shape
    lags = np.arange(L, dtype=np.float64)

    feature_groups = [infer_feature_group_refined(f) for f in feature_names]
    groups = sorted(set(feature_groups))
    group_counts = _group_counts_from_features(feature_names)

    rows = []
    lag_heat_total = []
    lag_heat_per_feature = []

    for group in groups:
        idx = [i for i, g in enumerate(feature_groups) if g == group]
        if not idx:
            continue

        sub = effective[:, idx, :]  # [D, n_features, L]

        total_l2 = float(np.linalg.norm(sub))
        per_feature_l2 = total_l2 / max(len(idx), 1)

        abs_lag_total = np.abs(sub).sum(axis=(0, 1))  # [L]
        abs_sum = max(float(abs_lag_total.sum()), 1e-12)
        center_lag = float((abs_lag_total * lags).sum() / abs_sum)
        argmax_lag = int(abs_lag_total.argmax())

        # Normalize each group's lag profile for visualization.
        total_profile = abs_lag_total / max(float(abs_lag_total.max()), 1e-12)
        per_feature_profile = total_profile / max(len(idx), 1)

        # For the per-feature heatmap, normalize after dividing, so color scale
        # still compares shapes, not just group size.
        per_feature_profile = per_feature_profile / max(float(per_feature_profile.max()), 1e-12)

        lag_heat_total.append(total_profile)
        lag_heat_per_feature.append(per_feature_profile)

        rows.append({
            "group": group,
            "num_features": len(idx),
            "total_effective_l2": total_l2,
            "mean_effective_l2_per_feature": per_feature_l2,
            "total_abs_lag_mass": float(abs_lag_total.sum()),
            "mean_abs_lag_mass_per_feature": float(abs_lag_total.sum()) / max(len(idx), 1),
            "abs_center_lag": center_lag,
            "argmax_abs_lag": argmax_lag,
            "features": "|".join(feature_names[i] for i in idx),
        })

    rows = sorted(rows, key=lambda r: r["total_effective_l2"], reverse=True)
    base._write_csv(tables / "feature_group_effective_temporal_summary.csv", rows)

    labels = [r["group"] for r in rows]

    _save_barh(
        figs / "feature_group_total_effective_l2.png",
        labels,
        np.asarray([r["total_effective_l2"] for r in rows]),
        title="Feature group total effective temporal loading",
        xlabel="total ||effective loading||2",
    )

    rows_by_avg = sorted(rows, key=lambda r: r["mean_effective_l2_per_feature"], reverse=True)
    _save_barh(
        figs / "feature_group_mean_effective_l2_per_feature.png",
        [r["group"] for r in rows_by_avg],
        np.asarray([r["mean_effective_l2_per_feature"] for r in rows_by_avg]),
        title="Feature group mean effective loading per feature",
        xlabel="mean ||effective loading||2 per feature",
    )

    rows_by_lag = sorted(rows, key=lambda r: r["abs_center_lag"], reverse=True)
    _save_barh(
        figs / "feature_group_center_lag.png",
        [r["group"] for r in rows_by_lag],
        np.asarray([r["abs_center_lag"] for r in rows_by_lag]),
        title="Feature group effective center lag",
        xlabel="abs-weighted center lag",
    )

    # Reconstruct heatmaps in sorted row order.
    group_to_total_profile = {}
    group_to_per_feature_profile = {}

    for group, total_profile, per_feature_profile in zip(groups, lag_heat_total, lag_heat_per_feature):
        group_to_total_profile[group] = total_profile
        group_to_per_feature_profile[group] = per_feature_profile

    group_order = [r["group"] for r in rows]

    heat_total = np.asarray([group_to_total_profile[g] for g in group_order], dtype=np.float64)
    heat_per_feature = np.asarray([group_to_per_feature_profile[g] for g in group_order], dtype=np.float64)

    _save_heatmap(
        figs / "feature_group_lag_response_heatmap_total_norm.png",
        heat_total,
        title="Feature group normalized |effective lag response|, total group mass",
        xlabel="lag",
        ylabel="group",
        xticklabels=[str(i) for i in range(L)],
        yticklabels=group_order,
    )

    _save_heatmap(
        figs / "feature_group_lag_response_heatmap_per_feature_norm.png",
        heat_per_feature,
        title="Feature group normalized |effective lag response|, per-feature adjusted",
        xlabel="lag",
        ylabel="group",
        xticklabels=[str(i) for i in range(L)],
        yticklabels=group_order,
    )

    return {
        "status": "ok",
        "groups": rows,
        "group_counts": group_counts,
        "meta": meta,
    }


# ============================================================
# Refined hidden channel interpretation
# ============================================================

def analyze_hidden_channels_refined(
    *,
    model,
    feature_names: list[str],
    output_dir: Path,
    top_terms: int,
) -> dict[str, Any]:
    tables = output_dir / "tables"
    figs = output_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    projection = base._get_input_projection(model)
    W = projection.weight.detach().float().cpu().numpy()

    bias = None
    if getattr(projection, "bias", None) is not None:
        bias = projection.bias.detach().float().cpu().numpy()

    # decode_projection_columns uses base.infer_feature_group through base.infer_feature_tags.
    decoded = base.decode_projection_columns(model, feature_names)

    group_feature_counts = _group_counts_from_features(feature_names)
    groups_all = sorted(group_feature_counts.keys())

    # Add categorical_or_extra if present.
    for info in decoded:
        if info["group"] not in group_feature_counts:
            group_feature_counts[info["group"]] = 1
            if info["group"] not in groups_all:
                groups_all.append(info["group"])

    groups_all = sorted(groups_all)

    rows = []

    for d in range(W.shape[0]):
        row_w = W[d]
        abs_w = np.abs(row_w)
        total_mass = float(abs_w.sum())

        group_mass: dict[str, float] = defaultdict(float)
        span_mass: dict[str, float] = defaultdict(float)

        for c, val in enumerate(row_w):
            info = decoded[c]
            group = str(info["group"])
            group_mass[group] += abs(float(val))

            if info["kind"] == "cnn":
                span_key = f"span{info['span']}"
            else:
                span_key = str(info["kind"])

            span_mass[span_key] += abs(float(val))

        group_avg_mass = {
            g: group_mass[g] / max(group_feature_counts.get(g, 1), 1)
            for g in group_mass
        }

        dominant_group_total = max(group_mass.items(), key=lambda kv: kv[1])[0]
        dominant_group_avg = max(group_avg_mass.items(), key=lambda kv: kv[1])[0]
        dominant_span = max(span_mass.items(), key=lambda kv: kv[1])[0]

        group_mass_values = np.asarray([group_mass.get(g, 0.0) for g in groups_all], dtype=np.float64)
        group_entropy = _entropy_from_mass(group_mass_values)
        group_entropy_norm = group_entropy / max(math.log(max(len(groups_all), 2)), 1e-12)

        pos_idx = np.argsort(row_w)[::-1][:top_terms]
        neg_idx = np.argsort(row_w)[:top_terms]

        def fmt(cols: np.ndarray) -> tuple[str, str, str, str]:
            terms = []
            vals = []
            groups = []
            spans = []

            for c in cols:
                c = int(c)
                info = decoded[c]
                kind = info["kind"]
                feature = info["feature"]
                group = info["group"]

                if kind == "cnn":
                    span = f"span{info['span']}"
                    term = f"{span}:{feature}"
                elif kind == "raw":
                    span = "raw"
                    term = f"raw:{feature}"
                else:
                    span = kind
                    term = f"{kind}:{feature}"

                terms.append(term)
                vals.append(f"{row_w[c]:.6g}")
                groups.append(str(group))
                spans.append(str(span))

            return "|".join(terms), "|".join(vals), "|".join(groups), "|".join(spans)

        pos_terms, pos_vals, pos_groups, pos_spans = fmt(pos_idx)
        neg_terms, neg_vals, neg_groups, neg_spans = fmt(neg_idx)

        rows.append({
            "hidden_dim": d,
            "weight_l2": float(np.linalg.norm(row_w)),
            "weight_l1": total_mass,
            "bias": float(bias[d]) if bias is not None else 0.0,

            "dominant_group_total": dominant_group_total,
            "dominant_group_total_mass": float(group_mass[dominant_group_total]),
            "dominant_group_total_mass_ratio": _safe_ratio(group_mass[dominant_group_total], total_mass),

            "dominant_group_avg_per_feature": dominant_group_avg,
            "dominant_group_avg_per_feature_mass": float(group_avg_mass[dominant_group_avg]),

            "group_entropy": group_entropy,
            "group_entropy_norm": group_entropy_norm,

            "dominant_span_or_branch": dominant_span,
            "dominant_span_mass": float(span_mass[dominant_span]),
            "dominant_span_mass_ratio": _safe_ratio(span_mass[dominant_span], total_mass),

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
    base._write_csv(tables / "hidden_channel_interpretation.csv", rows)

    # Heatmap for top hidden channels by weight_l2.
    top_rows = rows[:min(80, len(rows))]
    hidden_ids = [int(r["hidden_dim"]) for r in top_rows]

    heat_total = np.zeros((len(hidden_ids), len(groups_all)), dtype=np.float64)
    heat_per_feature = np.zeros_like(heat_total)

    for ii, d in enumerate(hidden_ids):
        row_w = W[d]
        group_mass: dict[str, float] = defaultdict(float)

        for c, val in enumerate(row_w):
            info = decoded[c]
            group_mass[str(info["group"])] += abs(float(val))

        for jj, g in enumerate(groups_all):
            heat_total[ii, jj] = group_mass.get(g, 0.0)
            heat_per_feature[ii, jj] = group_mass.get(g, 0.0) / max(group_feature_counts.get(g, 1), 1)

        # Row-normalize for visual comparison.
        if heat_total[ii].max() > 0:
            heat_total[ii] /= heat_total[ii].max()
        if heat_per_feature[ii].max() > 0:
            heat_per_feature[ii] /= heat_per_feature[ii].max()

    _save_heatmap(
        figs / "hidden_channel_group_mass_heatmap_top_total.png",
        heat_total,
        title="Top hidden channels: normalized group loading mass, total",
        xlabel="feature group",
        ylabel="hidden dim",
        xticklabels=groups_all,
        yticklabels=[str(d) for d in hidden_ids],
    )

    _save_heatmap(
        figs / "hidden_channel_group_mass_heatmap_top_per_feature.png",
        heat_per_feature,
        title="Top hidden channels: normalized group loading mass, per-feature adjusted",
        xlabel="feature group",
        ylabel="hidden dim",
        xticklabels=groups_all,
        yticklabels=[str(d) for d in hidden_ids],
    )

    # Dominant group distributions.
    total_counts: dict[str, int] = defaultdict(int)
    avg_counts: dict[str, int] = defaultdict(int)
    span_counts: dict[str, int] = defaultdict(int)

    for r in rows:
        total_counts[str(r["dominant_group_total"])] += 1
        avg_counts[str(r["dominant_group_avg_per_feature"])] += 1
        span_counts[str(r["dominant_span_or_branch"])] += 1

    dist_rows = []
    for g in sorted(set(total_counts) | set(avg_counts)):
        dist_rows.append({
            "group": g,
            "dominant_total_count": total_counts.get(g, 0),
            "dominant_avg_per_feature_count": avg_counts.get(g, 0),
            "num_features": group_feature_counts.get(g, 0),
        })
    base._write_csv(tables / "hidden_channel_dominant_group_distribution.csv", dist_rows)

    span_rows = [
        {"span_or_branch": k, "count": v}
        for k, v in sorted(span_counts.items(), key=lambda kv: kv[1], reverse=True)
    ]
    base._write_csv(tables / "hidden_channel_dominant_span_distribution.csv", span_rows)

    return {
        "status": "ok",
        "top_hidden_channels": rows[:20],
        "dominant_group_total_counts": dict(total_counts),
        "dominant_group_avg_per_feature_counts": dict(avg_counts),
        "dominant_span_counts": dict(span_counts),
    }


# ============================================================
# Patch base module and run
# ============================================================

def patch_base_module() -> None:
    base.infer_feature_group = infer_feature_group_refined
    base.infer_feature_tags = infer_feature_tags_refined

    base.analyze_feature_groups = analyze_feature_groups_refined
    base.analyze_hidden_channels = analyze_hidden_channels_refined


if __name__ == "__main__":
    patch_base_module()
    base.main()