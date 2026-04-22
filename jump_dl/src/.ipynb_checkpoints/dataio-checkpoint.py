from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader

__all__ = [
    "SliceBatchDataset",
    "SliceBatchDataLoader",
    "build_slice_dataloader",
    "SymbolDaySliceDataset",
    "symbol_day_pad_collate_fn",
    "build_symbol_day_index_cache",
]


def _identity_collate(x):
    # Dataset.__getitems__(list[int]) 已经直接返回一个完整 batch，
    # 所以 collate_fn 不需要再做任何事。
    return x


def _as_float_tensor(arr: np.ndarray, share_memory: bool = False) -> torch.Tensor:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.ascontiguousarray(arr)
    t = torch.from_numpy(arr)
    if share_memory:
        t = t.share_memory_()
    return t


class SliceBatchDataset(Dataset):
    """
    每个 sample = (date, symbol) 对应的一段连续时间序列。
    关键优化：
    1. 预先计算所有 sample 的 start/end 边界
    2. 实现 __getitems__(indices) 走 batched fetch
    3. 底层特征/标签预先 materialize 为 tensor
    """

    def __init__(
        self,
        df: pl.DataFrame,
        feature_cols: Mapping[str, Sequence[str]],
        target_cols: Sequence[str] = ("ret_30min",),
        symbol_col: str = "Symbol",
        time_col: str = "Time",
        sort: bool = True,
        share_memory: bool = True,
        return_meta: bool = True,
    ):
        self.feature_cols = OrderedDict((k, list(v)) for k, v in feature_cols.items())
        self.target_cols = list(target_cols)
        self.symbol_col = symbol_col
        self.time_col = time_col
        self.return_meta = return_meta

        work_df = df

        # 假设 time_col 是 Datetime；如果不是，你可以自己先加一个 date 列再改这里
        work_df = work_df.with_columns(
            pl.col(self.time_col).dt.date().alias("__date__")
        )

        if sort:
            work_df = work_df.sort(["__date__", self.symbol_col, self.time_col])

        # 排完序以后再打 row index，保证边界真的是连续 slice
        work_df = work_df.with_row_index("__row_idx__")

        spans = (
            work_df
            .group_by(["__date__", self.symbol_col], maintain_order=True)
            .agg(
                [
                    pl.col("__row_idx__").min().alias("start"),
                    (pl.col("__row_idx__").max() + 1).alias("end"),
                    pl.len().alias("length"),
                ]
            )
        )

        self.starts = spans["start"].to_numpy().astype(np.int64, copy=False)
        self.ends = spans["end"].to_numpy().astype(np.int64, copy=False)
        self.lengths = spans["length"].to_numpy().astype(np.int64, copy=False)

        # 去掉辅助列，保留排序后的主表
        work_df = work_df.drop(["__date__", "__row_idx__"])
        self.df = work_df

        # 预先 materialize 为 tensor，worker 下会明显更快
        self.feature_tensors: dict[str, torch.Tensor] = {}
        for name, cols in self.feature_cols.items():
            arr = work_df.select(cols).to_numpy()
            self.feature_tensors[name] = _as_float_tensor(arr, share_memory=share_memory)

        target_arr = work_df.select(self.target_cols).to_numpy()
        self.target_tensor = _as_float_tensor(target_arr, share_memory=share_memory)

        # meta 不参与 pin_memory，保留成 numpy/object 即可
        if self.return_meta:
            self.time_values = work_df.get_column(self.time_col).to_numpy()
            self.symbol_values = work_df.get_column(self.symbol_col).to_numpy()
        else:
            self.time_values = None
            self.symbol_values = None

    def __len__(self) -> int:
        return len(self.starts)

    def _normalize_indices(self, indices: Any) -> np.ndarray:
        if isinstance(indices, slice):
            idx = np.arange(*indices.indices(len(self)), dtype=np.int64)
        elif torch.is_tensor(indices):
            idx = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        else:
            idx = np.asarray(indices, dtype=np.int64)

        if idx.ndim == 0:
            idx = idx[None]
        return idx

    @staticmethod
    def _build_flat_positions(starts: np.ndarray, lengths: np.ndarray):
        """
        对于 batch 内变长序列，构造：
        - row_ids: 从底层大表里取哪些行
        - batch_ids / time_ids: 放到 padded batch 的哪里
        全是向量化构造，不用 Python 循环逐 sample copy。
        """
        total = int(lengths.sum())
        if total == 0:
            empty = np.empty((0,), dtype=np.int64)
            return empty, empty, empty

        batch_ids = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
        starts_rep = np.repeat(starts, lengths)

        group_offsets = np.repeat(np.cumsum(lengths) - lengths, lengths)
        time_ids = np.arange(total, dtype=np.int64) - group_offsets
        row_ids = starts_rep + time_ids
        return row_ids, batch_ids, time_ids

    def _build_batch(self, indices: Any) -> dict[str, Any]:
        idx = self._normalize_indices(indices)

        starts = self.starts[idx]
        ends = self.ends[idx]
        lengths = self.lengths[idx]

        B = len(idx)
        T = int(lengths.max()) if B > 0 else 0

        row_ids_np, batch_ids_np, time_ids_np = self._build_flat_positions(starts, lengths)

        row_ids = torch.from_numpy(row_ids_np).long()
        batch_ids = torch.from_numpy(batch_ids_np).long()
        time_ids = torch.from_numpy(time_ids_np).long()

        features: dict[str, torch.Tensor] = {}
        for name, base in self.feature_tensors.items():
            D = base.shape[1]
            out = torch.zeros((B, T, D), dtype=base.dtype)
            if row_ids.numel() > 0:
                out[batch_ids, time_ids] = base[row_ids]
            features[name] = out

        target_dim = self.target_tensor.shape[1]
        targets = torch.zeros((B, T, target_dim), dtype=self.target_tensor.dtype)
        if row_ids.numel() > 0:
            targets[batch_ids, time_ids] = self.target_tensor[row_ids]

        lengths_t = torch.from_numpy(lengths.copy()).long()
        padding_mask = torch.arange(T).unsqueeze(0) < lengths_t.unsqueeze(1)

        out: dict[str, Any] = {
            "features": features,
            "targets": targets,
            "lengths": lengths_t,
            "padding_mask": padding_mask,
        }

        if self.return_meta:
            # meta 这里保留成 list，不参与 pin_memory
            out["meta"] = [
                {
                    self.time_col: self.time_values[s:e],
                    self.symbol_col: self.symbol_values[s:e],
                    "start": int(s),
                    "end": int(e),
                }
                for s, e in zip(starts.tolist(), ends.tolist())
            ]

        return out

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # 单样本调试用
        return self._build_batch([idx])

    def __getitems__(self, indices: Sequence[int]) -> dict[str, Any]:
        """
        这是关键：
        DataLoader 在 auto-collation 模式下会优先走这个 batched fast-path
        （较新版本 PyTorch 支持）。
        """
        return self._build_batch(indices)


class SliceBatchDataLoader(DataLoader):
    def __init__(
        self,
        dataset: SliceBatchDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 0,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        **kwargs,
    ):
        loader_kwargs = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=_identity_collate,
            **kwargs,
        )

        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor
            loader_kwargs["persistent_workers"] = persistent_workers
        else:
            loader_kwargs["persistent_workers"] = False

        super().__init__(**loader_kwargs)


def build_slice_dataloader(
    df: pl.DataFrame,
    feature_cols: Mapping[str, Sequence[str]],
    target_cols: Sequence[str] = ("ret_30min",),
    symbol_col: str = "Symbol",
    time_col: str = "Time",
    batch_size: int = 32,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    sort: bool = True,
    share_memory: bool | None = True,
    return_meta: bool = True,
):
    """
    便捷入口。
    - Linux + num_workers>0: 建议 share_memory=True
    - 单进程/小数据: share_memory=False 也没问题
    """
    if share_memory is None:
        share_memory = num_workers > 0

    dataset = SliceBatchDataset(
        df=df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        symbol_col=symbol_col,
        time_col=time_col,
        sort=sort,
        share_memory=share_memory,
        return_meta=return_meta,
    )

    loader = SliceBatchDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )
    return loader


def build_symbol_day_index_cache(
    data_path: str | Path,
    *,
    sorted_data_path: str | Path,
    index_cache_path: str | Path,
    time_col: str = "Time",
    symbol_col: str = "Symbol",
    overwrite: bool = False,
) -> dict[str, Any]:
    sorted_path = Path(sorted_data_path)
    index_path = Path(index_cache_path)
    if not overwrite and (sorted_path.exists() or index_path.exists()):
        raise FileExistsError("Output exists. Pass overwrite=True to replace existing artifacts.")

    df = pl.read_parquet(data_path).with_columns(
        pl.col(time_col).dt.date().alias("__date__")
    ).sort(["__date__", symbol_col, time_col])
    df.write_parquet(sorted_path)

    indexed = df.with_row_index("__row_idx__")
    spans = (
        indexed.group_by(["__date__", symbol_col], maintain_order=True)
        .agg(
            pl.col("__row_idx__").min().alias("start"),
            (pl.col("__row_idx__").max() + 1).alias("end"),
            pl.len().alias("length"),
            pl.col(time_col).min().alias("time_min"),
            pl.col(time_col).max().alias("time_max"),
        )
        .rename({"__date__": "Date", symbol_col: "Symbol"})
    )
    spans.write_parquet(index_path)
    return {
        "sorted_data_path": str(sorted_path),
        "index_cache_path": str(index_path),
        "num_rows": df.height,
        "num_slices": spans.height,
    }


# Backward-compatible aliases for older training scripts.
SymbolDaySliceDataset = SliceBatchDataset
symbol_day_pad_collate_fn = _identity_collate
