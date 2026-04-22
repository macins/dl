# jump_dl

`jump_dl` 现在使用的是新版 slice-batch `dataio`。

核心约定：
- 一条 sample 表示某个 `Symbol` 在某一天里的完整日内序列
- dataloader 会按 sample 组 batch，并自动做右侧 padding
- batch 结构是：
  - `features`: `dict[str, Tensor[B, T, D]]`
  - `targets`: `Tensor[B, T, K]`
  - `lengths`: `Tensor[B]`
  - `padding_mask`: `Tensor[B, T]`
  - `meta`: `list[dict]`，可选
- 训练默认用 `padding_mask` 过滤 padding 位
- 当前 baseline 目标列是 `ret_30min`

## Quick Start

单次训练：

```bash
python jump_dl/scripts/run_train.py --config jump_dl/configs/base/train_gru.yaml
```

rolling 训练：

```bash
python jump_dl/scripts/run_rolling_train.py --config jump_dl/configs/base/train_gru_rolling.yaml
```

## Config Shape

新版配置里，`data.train_dataset` / `data.val_dataset` 主要关心这些字段：

```yaml
data:
  train_dataset:
    data_path: sample.sorted.parquet
    target_col: ret_30min
    time_col: Time
    symbol_col: Symbol
    feature_cols:
      continuous:
        - feature_imb_vol_2min
    categorical_cols: []
    vocab_path: null
    sort: true
    return_meta: true
```

说明：
- `data_path` 是实际读取的 parquet
- `feature_cols` 对应新版 `build_slice_dataloader(...)` 的 `feature_cols`
- `target_col` 会被脚本转成 `target_cols=[target_col]`
- `categorical_cols` 仅用于 vocab 编码；如果这些列已经是整数 id，也可以不配 `vocab_path`
- `sort=true` 时会按 `date -> symbol -> time` 在 dataloader 内重新整理 sample
- `return_meta=true` 时 batch 会附带每个 sample 的时间和 symbol 元信息

如果你不写 `feature_cols`，训练脚本会自动推断：
- 排除 `time_col`、`symbol_col`、目标列
- 排除 `categorical_cols`
- 剩下的数值列会放进模型的第一个 numeric feature group

如果模型有多个 numeric feature group，建议显式写 `feature_cols`，不要依赖自动推断。

## Dataset Filters

训练脚本当前支持这些 dataset 过滤字段：
- `start_date`
- `end_date`
- `years`
- `symbols`

rolling 训练就是通过给每个 fold 注入不同的 `start_date / end_date` 来完成切分的。

## Dataloader Shape

以单目标为例：

```python
batch = {
    "features": {
        "continuous": Tensor[B, T, D_num],
        "category": Tensor[B, T, D_cat],  # 如果有类别列
    },
    "targets": Tensor[B, T, 1],
    "lengths": Tensor[B],
    "padding_mask": Tensor[B, T],
    "meta": [...],  # optional
}
```

当前 `CosineSimilarityObjective` 和 `Trainer` 已经兼容这套结构。

## Build Slice Index

如果你还想生成排序后的 parquet 和 `(Date, Symbol)` 级别的 slice cache，可以跑：

```bash
python jump_dl/scripts/build_slice_index.py ^
  --data-path full_dataset.parquet ^
  --sorted-data-path full_dataset.sorted.parquet ^
  --index-cache-path full_dataset.symbol_day_index.parquet ^
  --overwrite
```

这个 index 文件现在主要是离线产物，训练脚本本身不依赖它。

## Categorical Vocab

如果有类别列，可以先生成 vocab：

```bash
python jump_dl/scripts/build_categorical_vocab.py ^
  --data-path full_dataset.parquet ^
  --categorical-cols feature_09,feature_10,feature_11 ^
  --output-path jump_dl/artifacts/full_vocab.json
```

训练脚本会把原值映射成：
- `0`: padding / null
- `1`: unknown
- `2...`: vocab 内已知值

## Notes

当前仓库里还保留了一些旧脚本名和旧产物名，但 `jump_dl` 训练主链路已经切到新版 `dataio` 了。配配置时优先参考 `train_gru.yaml` 和 `train_gru_rolling.yaml` 这两份示例。
