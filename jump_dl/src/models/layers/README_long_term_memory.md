# Long-Term Memory 模块说明

本文档介绍新加入的 `long_term_memory` 模块（`jump_dl/src/models/layers/long_term_memory.py`）的设计目标、模型结构和数据流（data flow）。

## 1. 设计目标

该模块用于在**日内时序 backbone** 之外，提供一个慢变化的长期上下文通道：

- 使用可学习的持久记忆 token（market / symbol）提供长期状态基底；
- 可选读取外部预计算的 rolling memory（例如过去若干交易日统计摘要）；
- 以最小侵入方式接入：第一版仅在 `pre_head`（backbone 后、预测头前）读取；
- 默认保持 baseline 行为不变：`long_term_memory.enabled=false` 时路径完全不启用。

---

## 2. 模块组成

`long_term_memory.py` 主要包含 3 个类：

1. `PersistentMemoryBank`
2. `PrecomputedMemoryEncoder`
3. `LongTermMemoryRead`

### 2.1 PersistentMemoryBank

作用：维护可学习 memory 参数，并按 batch 形状构造 `[B, N, K, D]` memory tokens。

- `market_memory`: `[num_market_slots, D]`（广播到所有 `B,N`）
- `symbol_memory`: `[num_symbols, num_symbol_slots, D]`（按 `symbol_ids` gather）

输出：
- `get_memory(...) -> [B, N, K_total, D]`

其中 `K_total = K_market + K_symbol`（按开启的 level 而定）。

### 2.2 PrecomputedMemoryEncoder

作用：把外部传入的 rolling memory 对齐/编码到 `[B, N, K, D]`。

支持输入形状：

- `[B, N, L, S]`：symbol 级摘要（`S -> D` 编码）
- `[B, L, S]`：market 级摘要（编码后 broadcast 到 N）
- `[B, N, K, D]`：已 token 化，直接使用
- `[B, K, D]`：已 token 化 market 级，broadcast 到 N

> 注意：本模块**不在 forward 中计算 rolling summary**，只消费 batch 提供的预计算结果。

### 2.3 LongTermMemoryRead

作用：对 hidden state `h=[B,N,T,D]` 做每个 symbol 独立的跨注意力读取（cross-attn）并残差注入。

核心步骤：

1. 组装 memory：
   - persistent memory（可选）
   - precomputed memory（可选）
   - 沿 slot 维拼接得到 `[B,N,K_total,D]`
2. query 处理：
   - `q = LN(h)`（可配置关闭）
3. 跨注意力读取：
   - 展平为 `Q=[B*N,T,D]`，`K/V=[B*N,K_total,D]`
   - 使用 `nn.MultiheadAttention(batch_first=True)`
4. 门控与残差：
   - `gate = sigmoid(W(LN(h)))`（scalar 或 vector）
   - `out = h + residual_alpha * dropout(gate * z)`

其中 `residual_alpha` 是可学习参数，初始化由 `residual_init` 控制。

---

## 3. 模型结构（与主模型集成位置）

在 `SequenceRegressor` 中，当前集成位置为：

```text
encoder -> backbone -> (optional symbol_query_decoder) -> (optional long_term_memory_read) -> head
```

即第一版只实现 `placement=pre_head`。

当 `long_term_memory.enabled=false`：

```text
encoder -> backbone -> (optional symbol_query_decoder) -> head
```

保持原行为。

---

## 4. Data Flow（按 forward 时序）

以下以 `x = [B,N,T,D]` 为 backbone 输出说明。

### 4.1 配置关闭（默认）

1. 读取输入 batch 并编码；
2. backbone 生成 `x`；
3. 直接送入 head 预测。

### 4.2 `mode=persistent`

1. `PersistentMemoryBank.get_memory(...)` 产出 `memory_p=[B,N,Kp,D]`；
2. `LongTermMemoryRead(x, precomputed_memory=None)`；
3. cross-attn 读取后残差写回，得到 `x'=[B,N,T,D]`；
4. `head(x')` 生成预测。

### 4.3 `mode=precomputed_context`

1. 从 `batch[batch_key]` 读取 rolling memory；
2. `PrecomputedMemoryEncoder` 转成 `memory_c=[B,N,Kc,D]`；
3. `LongTermMemoryRead(x, precomputed_memory=...)`；
4. 输出 `x'` 后进入 head。

若缺失 `batch_key`，会报清晰错误，避免 silent failure。

### 4.4 `mode=persistent_plus_precomputed`

1. 同时构建 `memory_p` 与 `memory_c`；
2. 拼接为 `[B,N,Kp+Kc,D]`；
3. 单次 cross-attn 读取并残差注入；
4. 输出至 head。

---

## 5. 关键张量形状

- hidden/query: `h = [B,N,T,D]`
- memory tokens: `[B,N,K,D]`
- 注意力内部展平：
  - `Q = [B*N, T, D]`
  - `K/V = [B*N, K, D]`
- 输出：`out = [B,N,T,D]`（与输入一致）

---

## 6. 训练稳定性与 baseline 兼容

- `enabled=false`：不创建/不调用读记忆分支，保持原路径；
- `residual_init=0.0`：初始时 `residual_alpha=0`，记忆分支初始不改动主干表示；
- memory slot 小、读取为 cross-attn，不引入时序递归状态。

---

## 7. 泄漏（leakage）约束

- `persistent` token 为可学习参数，不直接依赖当前样本未来信息；
- `precomputed` 输入必须由外部保证仅来自当前交易日前历史（`day < current_day`）；
- 当前版本不实现 test-time memory update，不使用标签/误差在线写入 memory。

---

## 8. 当前版本边界与后续 TODO

当前版本：

- ✅ `pre_head` 放置方式
- ✅ persistent / precomputed / mixed 三种读取模式
- ✅ gate + residual alpha + aux stats

待扩展（未在本 PR 实现）：

- `placement=block`（每层 block 注入）
- stateful EMA memory
- Titans 风格 test-time neural memory
- learned memory write/update 机制

