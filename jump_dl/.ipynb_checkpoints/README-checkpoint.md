# jump_dl

`jump_dl` 是一套独立于 `new_dl` 的新 pipeline，面向你现在这类数据定义：

- 行索引语义是 `Time + Symbol`
- 一个 sample 是一个 `symbol` 在某一天的一整段日内 time-series
- batch 由不同 sample 拼起来
- 在最前面对齐
- 后面不足的位置 `pad 0`
- label 是 `ret_30min`
- metric 是 cosine similarity，而不是 weighted R2

## 1. 整体流程

full dataset 接入建议按这个顺序：

1. 先把原始 parquet 排序并建立 `(date, symbol) -> (offset, length)` 的 slice index
2. 基于 full dataset 逐年计算 feature / target 统计量
3. 基于 full dataset 计算 categorical vocab
4. 用 `run_train.py` 跑单次训练，或者用 `run_rolling_train.py` 跑 expanding window

## 2. 生成 Slice Index

训练时的 `Dataset` 不是按整表扫，而是先读一个 offline index cache：

- 每一行对应一个 sample
- sample = 某个 `Date` 下的某个 `Symbol`
- cache 里至少包括：
  - `Date`
  - `Year`
  - `Symbol`
  - `symbol_id`
  - `offset`
  - `length`
  - `time_min`
  - `time_max`

先运行：

```bash
python jump_dl/scripts/build_slice_index.py ^
  --data-path full_dataset.parquet ^
  --sorted-data-path full_dataset.sorted.parquet ^
  --index-cache-path full_dataset.symbol_day_index.parquet ^
  --overwrite
```

说明：

- `sorted_data_path` 是训练实际读取的 parquet
- `index_cache_path` 是 sample 级索引缓存
- 这个步骤通常只需要离线做一次

## 3. 计算逐年统计量

为了后面做 expanding window，这里统计量按“逐年”落盘。脚本会输出每个数值列每年的：

- `count`
- `sum`
- `sum_sq`
- `min`
- `max`
- `null_count`
- `mean`
- `std`

运行方式：

```bash
python jump_dl/scripts/compute_feature_target_stats.py ^
  --data-path full_dataset.parquet ^
  --output-path jump_dl/artifacts/full_yearly_stats.parquet ^
  --time-col Time ^
  --symbol-col Symbol ^
  --target-col ret_30min
```

如果你不传 `--columns`，脚本会自动挑出数值列，并排除 `Time` / `Symbol`。

### 为什么要逐年统计

因为 rolling train 用的是 expanding window，所以第 `k` 折的 train stats 不能偷看未来年份。

当前 `jump_dl` 的默认做法是：

- 先离线把 full dataset 的 yearly stats 全部算出来
- 到某一折训练时，再根据 train window 覆盖到的年份，把这些 yearly stats 合并

例如：

- fold_00 训练集到 `2016-12-31`
- 那么它会只用 `2013, 2014, 2015, 2016` 的 yearly stats

## 4. 计算 Categorical Vocab

如果你有 categorical 列，需要先基于 full dataset 生成 vocab：

```bash
python jump_dl/scripts/build_categorical_vocab.py ^
  --data-path full_dataset.parquet ^
  --categorical-cols feature_09,feature_10,feature_11 ^
  --output-path jump_dl/artifacts/full_vocab.json
```

产物里每个 categorical 列会包含：

- `padding_token_id = 0`
- `unknown_token_id = 1`
- `values`
- `counts`
- `token_to_id`
- `vocab_size`

训练时 dataset 会：

- 把缺失值映射到 `0`
- 把未见过的新值映射到 `1`
- 已知 vocab 从 `2` 开始编码

## 5. 单次训练

单次训练用：

```bash
uv run -m jump_dl.scripts.run_train --config jump_dl/configs/base/train_gru.yaml
```

关键配置示例：

```yaml
data:
  train_dataset:
    data_path: full_dataset.sorted.parquet
    index_cache_path: full_dataset.symbol_day_index.parquet
    target_col: ret_30min
    categorical_cols: [feature_09, feature_10, feature_11]
    stats_path: jump_dl/artifacts/full_yearly_stats.parquet
    stats_years: [2013, 2014, 2015, 2016, 2017]
    vocab_path: jump_dl/artifacts/full_vocab.json
```

`stats_years` 是你显式指定的训练统计量来源年份。

## 6. Expanding Window Rolling

rolling 训练入口：

```bash
python jump_dl/scripts/run_rolling_train.py --config jump_dl/configs/base/train_gru_rolling.yaml
```

配置示例：

```yaml
rolling:
  output_root: workdirs/jump_dl/rolling
  stats_mode: expanding_years
  folds:
    - name: fold_00
      train_start: "2013-09-16"
      train_end: "2016-12-31"
      val_start: "2017-01-01"
      val_end: "2017-12-31"
    - name: fold_01
      train_start: "2013-09-16"
      train_end: "2017-12-31"
      val_start: "2018-01-01"
      val_end: "2018-12-31"
```

### expanding stats 的逻辑

当 `rolling.stats_mode = expanding_years` 时：

- 每一折的 train dataset 会根据 `train_start/train_end` 自动推断年份范围
- 然后只合并这些年份的 yearly stats

比如：

- `train_start = 2013-09-16`
- `train_end = 2018-12-31`

则这一折默认使用：

- `stats_years = [2013, 2014, 2015, 2016, 2017, 2018]`

这就满足 expanding window 的“统计量也不能看到未来”的要求。

## 7. Dataset 过滤能力

`SymbolDaySliceDataset` 当前支持这些 sample 级过滤：

- `start_date`
- `end_date`
- `years`
- `symbols`

rolling train 就是基于这个机制做 train/val 切分的。

## 8. 当前 Metric 和 Objective

label:

- `ret_30min`

metric:

```text
mean(y * y_pred) / sqrt(mean(y ** 2) * mean(y_pred ** 2))
```

trainer 当前默认：

- loss = `1 - cosine_similarity`
- monitor key = `cosine_similarity`
- monitor mode = `max`

## 9. 当前产物位置

常见产物：

- `full_dataset.sorted.parquet`
- `full_dataset.symbol_day_index.parquet`
- `jump_dl/artifacts/full_yearly_stats.parquet`
- `jump_dl/artifacts/full_vocab.json`
- `workdirs/jump_dl/...`

## 10. 建议的 full dataset 实操顺序

建议按这个顺序跑：

1. `build_slice_index.py`
2. `compute_feature_target_stats.py`
3. `build_categorical_vocab.py`
4. 先跑一个单次 train smoke test
5. 再跑 `run_rolling_train.py`

如果你已经确定 rolling 的切分方案，最推荐直接维护一份 rolling config，把每折的 train/val 时间边界写清楚。
