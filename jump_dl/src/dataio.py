from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader

import pickle

__all__ = [
    "SliceBatchDataset",
    "SliceBatchDataLoader",
    "build_slice_dataloader",

    # new
    "MarketDayDataset",
    "MarketDayDataLoader",
    "build_market_day_dataloader",

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

        with open("/root/autodl-tmp/dl/jump_dl/artifacts/vocabs.pkl", "rb") as f:
            vocabs = pickle.load(f)

        for col in vocabs:
            df = df.with_columns([
                pl.col(col).replace_strict(vocabs[col]).alias(col)
            ])

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
            "target_col_names": list(self.target_cols),
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

class MarketDayDataset(Dataset):
    """
    每个 sample = 一个交易日的全市场张量。

    Output batch:
        features[group]: (B, N, T, D)
        targets:         (B, N, T, Y)
        padding_mask:    (B, N, T), True 表示这个 symbol/time 有真实 row
        symbol_mask:     (B, N),    True 表示这个 symbol slot 有效
        time_mask:       (B, T),    True 表示这个 time slot 有效

    这里 N/T 可以按 batch 内最大值 padding。
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
        validate_unique: bool = True,
        vocab_path: str | Path | None = "/root/autodl-tmp/dl/jump_dl/artifacts/vocabs.pkl",
    ):
        self.feature_cols = OrderedDict((k, list(v)) for k, v in feature_cols.items())
        self.target_cols = list(target_cols)
        self.symbol_col = str(symbol_col)
        self.time_col = str(time_col)
        self.return_meta = bool(return_meta)

        if vocab_path is not None and Path(vocab_path).exists():
            with open(vocab_path, "rb") as f:
                vocabs = pickle.load(f)

            for col in vocabs:
                if col in df.columns:
                    df = df.with_columns(
                        pl.col(col).replace_strict(vocabs[col]).alias(col)
                    )

        work_df = df.with_columns(
            pl.col(self.time_col).dt.date().alias("__date__")
        )

        if validate_unique:
            dup = (
                work_df
                .group_by(["__date__", self.symbol_col, self.time_col])
                .len()
                .filter(pl.col("len") > 1)
            )
            if dup.height > 0:
                raise ValueError(
                    "Found duplicate rows for (__date__, symbol, time). "
                    "MarketDayDataset expects at most one row per date/symbol/time. "
                    f"First duplicates:\n{dup.head(10)}"
                )

        if sort:
            work_df = work_df.sort(["__date__", self.time_col, self.symbol_col])

        # row_idx 是在排序后的主表上的位置；后面 feature_tensors/target_tensor 都按这个顺序 materialize。
        work_df = work_df.with_row_index("__row_idx__")

        self._build_day_records(work_df)

        # 去掉辅助列，保留排序后的主表
        tensor_df = work_df.drop(["__date__", "__row_idx__"])
        self.df = tensor_df

        self.feature_tensors: dict[str, torch.Tensor] = {}
        for name, cols in self.feature_cols.items():
            arr = tensor_df.select(cols).to_numpy()
            self.feature_tensors[name] = _as_float_tensor(arr, share_memory=share_memory)

        target_arr = tensor_df.select(self.target_cols).to_numpy()
        self.target_tensor = _as_float_tensor(target_arr, share_memory=share_memory)

    def _build_day_records(self, work_df: pl.DataFrame) -> None:
        """
        为每一天预先保存：
            row_ids:    这一日所有真实 row 在 tensor_df 里的 row index
            symbol_ids: 每个 row 对应 local N index
            time_ids:   每个 row 对应 local T index
            symbols:    local N index -> symbol value
            times:      local T index -> time value
        """
        self.records: list[dict[str, Any]] = []

        # partition_by 比 group_by agg(list) 更直接，便于构造 local id。
        parts = work_df.partition_by("__date__", maintain_order=True)

        for part in parts:
            date_value = part.get_column("__date__")[0]

            row_ids = part.get_column("__row_idx__").to_numpy().astype(np.int64, copy=False)
            symbols_arr = part.get_column(self.symbol_col).to_numpy()
            times_arr = part.get_column(self.time_col).to_numpy()

            # work_df 已经按 date/time/symbol 排序。
            # np.unique 会排序；对 symbol/time 来说这通常是稳定且可接受的。
            # 如果你想严格保留出现顺序，可以换成 _unique_list_preserve_order。
            unique_symbols = np.unique(symbols_arr)
            unique_times = np.unique(times_arr)

            symbol_to_idx = {v: i for i, v in enumerate(unique_symbols.tolist())}
            time_to_idx = {v: i for i, v in enumerate(unique_times.tolist())}

            symbol_ids = np.asarray(
                [symbol_to_idx[v] for v in symbols_arr.tolist()],
                dtype=np.int64,
            )
            time_ids = np.asarray(
                [time_to_idx[v] for v in times_arr.tolist()],
                dtype=np.int64,
            )

            self.records.append(
                {
                    "date": date_value,
                    "row_ids": row_ids,
                    "symbol_ids": symbol_ids,
                    "time_ids": time_ids,
                    "symbols": unique_symbols,
                    "times": unique_times,
                    "num_symbols": int(len(unique_symbols)),
                    "num_times": int(len(unique_times)),
                }
            )

    def __len__(self) -> int:
        return len(self.records)

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

    def _build_batch(self, indices: Any) -> dict[str, Any]:
        idx = self._normalize_indices(indices)
        selected = [self.records[int(i)] for i in idx.tolist()]

        B = len(selected)
        N = max((r["num_symbols"] for r in selected), default=0)
        T = max((r["num_times"] for r in selected), default=0)

        # 每条真实 row 放到 batch 里的位置
        batch_ids_np: list[np.ndarray] = []
        symbol_ids_np: list[np.ndarray] = []
        time_ids_np: list[np.ndarray] = []
        row_ids_np: list[np.ndarray] = []

        for b, rec in enumerate(selected):
            n_rows = len(rec["row_ids"])
            batch_ids_np.append(np.full(n_rows, b, dtype=np.int64))
            symbol_ids_np.append(rec["symbol_ids"])
            time_ids_np.append(rec["time_ids"])
            row_ids_np.append(rec["row_ids"])

        if B == 0:
            batch_ids = torch.empty((0,), dtype=torch.long)
            symbol_ids = torch.empty((0,), dtype=torch.long)
            time_ids = torch.empty((0,), dtype=torch.long)
            row_ids = torch.empty((0,), dtype=torch.long)
        else:
            batch_ids = torch.from_numpy(np.concatenate(batch_ids_np)).long()
            symbol_ids = torch.from_numpy(np.concatenate(symbol_ids_np)).long()
            time_ids = torch.from_numpy(np.concatenate(time_ids_np)).long()
            row_ids = torch.from_numpy(np.concatenate(row_ids_np)).long()

        features: dict[str, torch.Tensor] = {}
        for name, base in self.feature_tensors.items():
            D = base.shape[1]
            out = torch.zeros((B, N, T, D), dtype=base.dtype)
            if row_ids.numel() > 0:
                out[batch_ids, symbol_ids, time_ids] = base[row_ids]
            features[name] = out

        target_dim = self.target_tensor.shape[1]
        targets = torch.zeros((B, N, T, target_dim), dtype=self.target_tensor.dtype)
        if row_ids.numel() > 0:
            targets[batch_ids, symbol_ids, time_ids] = self.target_tensor[row_ids]

        padding_mask = torch.zeros((B, N, T), dtype=torch.bool)
        if row_ids.numel() > 0:
            padding_mask[batch_ids, symbol_ids, time_ids] = True

        symbol_lengths = torch.tensor(
            [r["num_symbols"] for r in selected],
            dtype=torch.long,
        )
        time_lengths = torch.tensor(
            [r["num_times"] for r in selected],
            dtype=torch.long,
        )

        symbol_mask = torch.arange(N).unsqueeze(0) < symbol_lengths.unsqueeze(1)
        time_mask = torch.arange(T).unsqueeze(0) < time_lengths.unsqueeze(1)

        out: dict[str, Any] = {
            "features": features,
            "targets": targets,
            "target_col_names": list(self.target_cols),

            # 兼容 trainer/objective 里常用的 mask 名字。
            # 注意这里 padding_mask 是 3D: (B, N, T)
            "padding_mask": padding_mask,

            # 更明确的名字
            "observed_mask": padding_mask,
            "symbol_mask": symbol_mask,
            "time_mask": time_mask,
            "symbol_lengths": symbol_lengths,
            "time_lengths": time_lengths,

            # backward-compatible，但这里表示 time length，不再是旧版 B 条 symbol 序列长度。
            "lengths": time_lengths,
        }

        if self.return_meta:
            out["meta"] = [
                {
                    "date": rec["date"],
                    self.symbol_col: rec["symbols"],
                    self.time_col: rec["times"],
                    "num_symbols": rec["num_symbols"],
                    "num_times": rec["num_times"],
                }
                for rec in selected
            ]

        return out

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._build_batch([idx])

    def __getitems__(self, indices: Sequence[int]) -> dict[str, Any]:
        return self._build_batch(indices)


class MarketDayDataLoader(DataLoader):
    def __init__(
        self,
        dataset: MarketDayDataset,
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


def build_market_day_dataloader(
    df: pl.DataFrame,
    feature_cols: Mapping[str, Sequence[str]],
    target_cols: Sequence[str] = ("ret_30min",),
    symbol_col: str = "Symbol",
    time_col: str = "Time",
    batch_size: int = 1,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    sort: bool = True,
    share_memory: bool | None = True,
    return_meta: bool = True,
    validate_unique: bool = True,
    vocab_path: str | Path | None = "/root/autodl-tmp/dl/jump_dl/artifacts/vocabs.pkl",
):
    """
    每个 sample 是一个 date 的全市场面板:
        features[group]: (B, N, T, D)

    建议 batch_size 先从 1 或 2 开始。
    """
    if share_memory is None:
        share_memory = num_workers > 0

    dataset = MarketDayDataset(
        df=df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        symbol_col=symbol_col,
        time_col=time_col,
        sort=sort,
        share_memory=share_memory,
        return_meta=return_meta,
        validate_unique=validate_unique,
        vocab_path=vocab_path,
    )

    loader = MarketDayDataLoader(
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
    
# Backward-compatible aliases for older training scripts.
SymbolDaySliceDataset = SliceBatchDataset
symbol_day_pad_collate_fn = _identity_collate
