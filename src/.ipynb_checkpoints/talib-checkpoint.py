import polars as pl
from typing import List, Optional
from functools import partial

# ========= 通用 helper =========

def _over(expr: pl.Expr, group_col: Optional[str]) -> pl.Expr:
    """按 Symbol 等分组做所有计算。group_col=None 时不分组。"""
    return expr.over(group_col) if group_col is not None else expr


def _price(col: str) -> pl.Expr:
    """
    价格类字段的统一清洗:
    - 转 float
    - NaN -> null
    - 前向填充
    - 开头仍为空的地方填成 1.0
    """
    return (
        pl.col(col)
        .cast(pl.Float64)
        .fill_nan(None)
        .fill_null(strategy="forward")
        .fill_null(1.0)
    )


def _volume_like(col: str) -> pl.Expr:
    """
    成交量 / 持仓量类字段的统一清洗:
    - 转 float
    - NaN/null 直接填 0.0
    """
    return (
        pl.col(col)
        .cast(pl.Float64)
        .fill_nan(0.0)
        .fill_null(0.0)
    )


def sma(col: str, window: int) -> pl.Expr:
    # 这里默认用于价格类
    return _price(col).rolling_mean(window_size=window, min_periods=window)


def stddev(col: str, window: int) -> pl.Expr:
    # 默认用于价格类
    return _price(col).rolling_std(window_size=window, min_periods=window)


def ema(col: str, span: int) -> pl.Expr:
    """使用标准 EMA alpha = 2/(span+1)，默认用于价格类。"""
    alpha = 2.0 / (span + 1.0)
    return _price(col).ewm_mean(alpha=alpha, adjust=False)


def typical_price() -> pl.Expr:
    # TP = (H + L + C) / 3
    return (_price("High") + _price("Low") + _price("Close")) / 3.0


def true_range() -> pl.Expr:
    """Wilder TR，基于清洗后的价格。"""
    high = _price("High")
    low = _price("Low")
    close_prev = _price("Close").shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    return pl.max_horizontal(tr1, tr2, tr3)


# 注意：下面所有函数都假定 df 已经按 [group_col, time_col] 排好序。


# ========= 1. MACD =========
def add_macd(
    lf: pl.LazyFrame,
    close: str = "Close",
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    group_col: str = "Symbol",
) -> pl.LazyFrame:
    dif = ema(close, fast) - ema(close, slow)
    alpha_signal = 2.0 / (signal + 1.0)
    dea = dif.ewm_mean(alpha=alpha_signal, adjust=False)
    bar = dif - dea

    return lf.with_columns([
        _over(dif, group_col).alias(f"feature_macd_dif_{fast}_{slow}"),
        _over(dea, group_col).alias(f"feature_macd_dea_{fast}_{slow}_{signal}"),
        _over(bar, group_col).alias(f"feature_macd_bar_{fast}_{slow}_{signal}"),
    ])


# ========= 2. DMI / ADX =========
def add_dmi(
    lf: pl.LazyFrame,
    windows: List[int],
    group_col: str = "Symbol",
) -> pl.LazyFrame:
    high = _price("High")
    low = _price("Low")

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pl.when((up_move > 0) & (up_move > down_move)).then(up_move).otherwise(0.0)
    minus_dm = pl.when((down_move > 0) & (down_move > up_move)).then(down_move).otherwise(0.0)

    tr = true_range()

    exprs = []
    eps = 1e-12
    for n in windows:
        alpha_n = 2.0 / (n + 1.0)
        atr = tr.ewm_mean(alpha=alpha_n, adjust=False)
        atr_safe = atr + eps

        plus_di = 100.0 * plus_dm.ewm_mean(alpha=alpha_n, adjust=False) / atr_safe
        minus_di = 100.0 * minus_dm.ewm_mean(alpha=alpha_n, adjust=False) / atr_safe

        denom = plus_di + minus_di
        dx = 100.0 * (plus_di - minus_di).abs() / (denom + eps)

        adx = dx.ewm_mean(alpha=alpha_n, adjust=False)
        adxr = (adx + adx.shift(n)) / 2.0

        exprs.extend([
            _over(plus_di, group_col).alias(f"feature_dmi_plus_di_{n}"),
            _over(minus_di, group_col).alias(f"feature_dmi_minus_di_{n}"),
            _over(adx, group_col).alias(f"feature_dmi_adx_{n}"),
            _over(adxr, group_col).alias(f"feature_dmi_adxr_{n}"),
        ])

    return lf.with_columns(exprs)


# ========= 3. DMA =========
def add_dma(
    lf: pl.LazyFrame,
    close: str = "Close",
    short: int = 10,
    long: int = 50,
    ama_window: int = 10,
    group_col: str = "Symbol",
) -> pl.LazyFrame:
    ma_short = sma(close, short)
    ma_long = sma(close, long)
    dma = ma_short - ma_long
    ama = dma.rolling_mean(window_size=ama_window, min_periods=ama_window)

    return lf.with_columns([
        _over(dma, group_col).alias(f"feature_dma_{short}_{long}"),
        _over(ama, group_col).alias(f"feature_dma_ama_{short}_{long}_{ama_window}"),
    ])


# ========= 4. EXPMA（多周期 EMA）=========

def add_expma(
    lf: pl.LazyFrame,
    windows: List[int],
    close: str = "Close",
    group_col: str = "Symbol",
) -> pl.LazyFrame:
    exprs = [
        _over(ema(close, n), group_col).alias(f"feature_expma_{n}")
        for n in windows
    ]
    return lf.with_columns(exprs)


# ========= 5. TRIX =========
def add_trix(
    lf: pl.LazyFrame,
    close: str = "Close",
    n: int = 12,
    signal: int = 9,
    group_col: str = "Symbol",
) -> pl.LazyFrame:
    alpha_n = 2.0 / (n + 1.0)
    alpha_sig = 2.0 / (signal + 1.0)

    ema1 = ema(close, n)
    ema2 = ema1.ewm_mean(alpha=alpha_n, adjust=False)
    ema3 = ema2.ewm_mean(alpha=alpha_n, adjust=False)

    eps = 1e-12
    trix_raw = (ema3 - ema3.shift(1)) / (ema3.shift(1) + eps) * 100.0
    trix_signal = trix_raw.ewm_mean(alpha=alpha_sig, adjust=False)

    return lf.with_columns([
        _over(trix_raw, group_col).alias(f"feature_trix_{n}"),
        _over(trix_signal, group_col).alias(f"feature_trix_signal_{n}_{signal}"),
    ])


# ========= 6. BR / AR =========
def add_brar(
    lf: pl.LazyFrame,
    windows: List[int],
    group_col: str = "Symbol",
) -> pl.LazyFrame:
    o = _price("Open")
    h = _price("High")
    l = _price("Low")
    c_prev = _price("Close").shift(1)

    ar_up = h - o
    ar_down = o - l

    br_up = pl.max_horizontal(h - c_prev, pl.lit(0.0))
    br_down = pl.max_horizontal(c_prev - l, pl.lit(0.0))

    exprs = []
    eps = 1e-12
    for n in windows:
        ar = 100.0 * ar_up.rolling_sum(n) / (ar_down.rolling_sum(n) + eps)
        br = 100.0 * br_up.rolling_sum(n) / (br_down.rolling_sum(n) + eps)

        exprs.extend([
            _over(ar, group_col).alias(f"feature_brar_ar_{n}"),
            _over(br, group_col).alias(f"feature_brar_br_{n}"),
        ])

    return lf.with_columns(exprs)


# ========= 7. CR =========
def add_cr(
    lf: pl.LazyFrame,
    windows: List[int],
    group_col: str = "Symbol",
) -> pl.LazyFrame:
    h = _price("High")
    l = _price("Low")

    mid_prev = (h.shift(1) + l.shift(1)) / 2.0

    up = pl.max_horizontal(h - mid_prev, pl.lit(0.0))
    down = pl.max_horizontal(mid_prev - l, pl.lit(0.0))

    exprs = []
    eps = 1e-12
    for n in windows:
        num = up.rolling_sum(n)
        den = down.rolling_sum(n)
        cr = 100.0 * num / (den + eps)
        exprs.append(_over(cr, group_col).alias(f"feature_cr_{n}"))

    return lf.with_columns(exprs)


# ========= 8. VR 成交量比率 =========
def add_vr(
    lf: pl.LazyFrame,
    windows: List[int],
    group_col: str = "Symbol",
) -> pl.LazyFrame:
    c = _price("Close")
    c_prev = c.shift(1)
    v = _volume_like("Volume")

    up_vol = pl.when(c > c_prev).then(v).otherwise(0.0)
    down_vol = pl.when(c < c_prev).then(v).otherwise(0.0)
    eq_vol = pl.when(c == c_prev).then(v).otherwise(0.0)

    exprs = []
    eps = 1e-12
    for n in windows:
        up_sum = up_vol.rolling_sum(n)
        down_sum = down_vol.rolling_sum(n)
        eq_sum = eq_vol.rolling_sum(n)

        vr = 100.0 * (up_sum + 0.5 * eq_sum) / (down_sum + 0.5 * eq_sum + eps)
        exprs.append(_over(vr, group_col).alias(f"feature_vr_{n}"))

    return lf.with_columns(exprs)


# ========= 9. OBV =========
def add_obv(
    lf: pl.LazyFrame,
    group_col: str = "Symbol",
) -> pl.LazyFrame:
    c = _price("Close")
    c_prev = c.shift(1)
    v = _volume_like("Volume")

    signed_vol = (
        pl.when(c > c_prev).then(v)
        .when(c < c_prev).then(-v)
        .otherwise(0.0)
    )

    obv = signed_vol.cum_sum()

    return lf.with_columns([
        _over(obv, group_col).alias("feature_obv"),
    ])


# ========= 13. RSI =========
def add_rsi(
    lf: pl.LazyFrame,
    windows: List[int],
    close: str = "Close",
    group_col: str = "Symbol",
) -> pl.LazyFrame:
    c = _price(close)
    delta = c - c.shift(1)
    gain = pl.when(delta > 0).then(delta).otherwise(0.0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0.0)

    exprs = []
    eps = 1e-12
    for n in windows:
        alpha_n = 2.0 / (n + 1.0)
        avg_gain = gain.ewm_mean(alpha=alpha_n, adjust=False)
        avg_loss = loss.ewm_mean(alpha=alpha_n, adjust=False)
        rs = avg_gain / (avg_loss + eps)
        rsi = 100.0 - 100.0 / (1.0 + rs)
        exprs.append(_over(rsi, group_col).alias(f"feature_rsi_{n}"))

    return lf.with_columns(exprs)


# ========= 14. W%R =========
def add_wpr(
    lf: pl.LazyFrame,
    windows: List[int],
    group_col: str = "Symbol",
) -> pl.LazyFrame:
    h = _price("High")
    l = _price("Low")
    c = _price("Close")

    exprs = []
    eps = 1e-12
    for n in windows:
        hh = h.rolling_max(window_size=n, min_periods=n)
        ll = l.rolling_min(window_size=n, min_periods=n)
        wpr = 100.0 - (c - ll) / (hh - ll + eps) * 100.0
        exprs.append(_over(wpr, group_col).alias(f"feature_wpr_{n}"))

    return lf.with_columns(exprs)


# ========= 16. KDJ =========
def add_kdj(
    lf: pl.LazyFrame,
    n: int = 9,
    m1: int = 3,
    m2: int = 3,
    group_col: str = "Symbol",
) -> pl.LazyFrame:
    h = _price("High")
    l = _price("Low")
    c = _price("Close")

    hh = h.rolling_max(window_size=n, min_periods=n)
    ll = l.rolling_min(window_size=n, min_periods=n)

    eps = 1e-12
    rsv = (c - ll) / (hh - ll + eps) * 100.0

    alpha_k = 2.0 / (m1 + 1.0)
    alpha_d = 2.0 / (m2 + 1.0)

    k = rsv.ewm_mean(alpha=alpha_k, adjust=False)
    d = k.ewm_mean(alpha=alpha_d, adjust=False)
    j = 3 * k - 2 * d

    return lf.with_columns([
        _over(k, group_col).alias(f"feature_kdj_k_{n}_{m1}_{m2}"),
        _over(d, group_col).alias(f"feature_kdj_d_{n}_{m1}_{m2}"),
        _over(j, group_col).alias(f"feature_kdj_j_{n}_{m1}_{m2}"),
    ])


# ========= 17. CCI =========
def add_cci(
    lf: pl.LazyFrame,
    windows: List[int],
    group_col: str = "Symbol",
) -> pl.LazyFrame:
    tp = typical_price()

    exprs = []
    eps = 1e-12
    for n in windows:
        ma = tp.rolling_mean(window_size=n, min_periods=n)
        md = (tp - ma).abs().rolling_mean(window_size=n, min_periods=n)
        cci = (tp - ma) / (0.015 * md + eps)
        exprs.append(_over(cci, group_col).alias(f"feature_cci_{n}"))

    return lf.with_columns(exprs)


# ========= 18. ROC =========
def add_roc(
    lf: pl.LazyFrame,
    windows: List[int],
    close: str = "Close",
    group_col: str = "Symbol",
) -> pl.LazyFrame:
    c = _price(close)
    exprs = []
    eps = 1e-12
    for n in windows:
        roc = (c - c.shift(n)) / (c.shift(n) + eps) * 100.0
        exprs.append(_over(roc, group_col).alias(f"feature_roc_{n}"))

    return lf.with_columns(exprs)


# ========= 20. Bollinger Bands =========
def add_bollinger(
    lf: pl.LazyFrame,
    windows: List[int],
    close: str = "Close",
    k: float = 2.0,
    group_col: str = "Symbol",
) -> pl.LazyFrame:
    exprs = []
    for n in windows:
        mid = sma(close, n)
        sd = stddev(close, n)
        upper = mid + k * sd
        lower = mid - k * sd

        exprs.extend([
            _over(mid, group_col).alias(f"feature_bollinger_mid_{n}"),
            _over(upper, group_col).alias(f"feature_bollinger_up_{n}"),
            _over(lower, group_col).alias(f"feature_bollinger_low_{n}"),
        ])

    return lf.with_columns(exprs)


# ========= 22. TURN 周转率（这里用 Volume / OpenInterest 近似）=========
def add_turn(
    lf: pl.LazyFrame,
    windows: List[int],
    group_col: str = "Symbol",
) -> pl.LazyFrame:
    v = _volume_like("Volume")
    oi = _volume_like("OpenInterest")

    exprs = []
    eps = 1e-12
    for n in windows:
        vol_sum = v.rolling_sum(window_size=n, min_periods=n)
        oi_mean = oi.rolling_mean(window_size=n, min_periods=n)
        turn = vol_sum / (oi_mean + eps)
        exprs.append(_over(turn, group_col).alias(f"feature_turn_{n}"))

    return lf.with_columns(exprs)


# ========= 23. BIAS 乖离率 =========
def add_bias(
    lf: pl.LazyFrame,
    windows: List[int],
    close: str = "Close",
    group_col: str = "Symbol",
) -> pl.LazyFrame:
    c = _price(close)
    exprs = []
    eps = 1e-12
    for n in windows:
        ma = sma(close, n)
        bias = (c - ma) / (ma + eps) * 100.0
        exprs.append(_over(bias, group_col).alias(f"feature_bias_{n}"))
    return lf.with_columns(exprs)


# ========= 27. MTM 动量 =========
def add_mtm(
    lf: pl.LazyFrame,
    windows: List[int],
    close: str = "Close",
    group_col: str = "Symbol",
) -> pl.LazyFrame:
    c = _price(close)
    exprs = []
    for n in windows:
        mtm = c - c.shift(n)
        exprs.append(_over(mtm, group_col).alias(f"feature_mtm_{n}"))
    return lf.with_columns(exprs)


# ========= 28. PSY 心理线 =========
def add_psy(
    lf: pl.LazyFrame,
    windows: List[int],
    close: str = "Close",
    group_col: str = "Symbol",
) -> pl.LazyFrame:
    c = _price(close)
    up_day = pl.when(c > c.shift(1)).then(1.0).otherwise(0.0)

    exprs = []
    for n in windows:
        psy = up_day.rolling_sum(window_size=n, min_periods=n) / float(n) * 100.0
        exprs.append(_over(psy, group_col).alias(f"feature_psy_{n}"))
    return lf.with_columns(exprs)


# ========= 29. OSC 摆动线 =========
def add_osc(
    lf: pl.LazyFrame,
    windows: List[int],
    close: str = "Close",
    group_col: str = "Symbol",
) -> pl.LazyFrame:
    c = _price(close)
    exprs = []
    for n in windows:
        ma = sma(close, n)
        osc = c - ma
        exprs.append(_over(osc, group_col).alias(f"feature_osc_{n}"))
    return lf.with_columns(exprs)


# ========= 30. B3612 三减六日乖离 =========
def add_b3612(
    lf: pl.LazyFrame,
    close: str = "Close",
    n1: int = 3,
    n2: int = 6,
    n3: int = 12,
    group_col: str = "Symbol",
) -> pl.LazyFrame:
    ma1 = sma(close, n1)
    ma2 = sma(close, n2)
    ma3 = sma(close, n3)

    b36 = ma1 - ma2
    b612 = ma2 - ma3

    return lf.with_columns([
        _over(b36, group_col).alias(f"feature_b36_{n1}_{n2}"),
        _over(b612, group_col).alias(f"feature_b612_{n2}_{n3}"),
    ])


# ========= 总入口：一键添加所有 TA 特征 =========
def add_ta(
    df: pl.DataFrame,
    group_col: str = "Symbol",
    time_col: str = "Time",
) -> pl.DataFrame:
    lf = df.sort([group_col, time_col]).lazy()

    funcs = [
        partial(add_macd, fast=12, slow=26, signal=9, group_col=group_col),
        partial(add_dmi, windows=[14], group_col=group_col),
        partial(add_dma, close="Close", short=10, long=50, ama_window=10, group_col=group_col),
        partial(add_expma, windows=[5, 10, 20, 60], close="Close", group_col=group_col),
        partial(add_trix, close="Close", n=12, signal=9, group_col=group_col),
        partial(add_brar, windows=[26], group_col=group_col),
        partial(add_cr, windows=[26], group_col=group_col),
        partial(add_vr, windows=[26], group_col=group_col),
        partial(add_obv, group_col=group_col),
        partial(add_rsi, windows=[6, 12, 24], close="Close", group_col=group_col),
        partial(add_wpr, windows=[14], group_col=group_col),
        partial(add_kdj, n=9, m1=3, m2=3, group_col=group_col),
        partial(add_cci, windows=[14], group_col=group_col),
        partial(add_roc, windows=[10], close="Close", group_col=group_col),
        partial(add_bollinger, windows=[20], close="Close", k=2.0, group_col=group_col),
        partial(add_turn, windows=[60], group_col=group_col),
        partial(add_bias, windows=[6, 12, 24, 72], close="Close", group_col=group_col),
        partial(add_mtm, windows=[10], close="Close", group_col=group_col),
        partial(add_psy, windows=[12], close="Close", group_col=group_col),
        partial(add_osc, windows=[6, 12, 24], close="Close", group_col=group_col),
        partial(add_b3612, close="Close", n1=3, n2=6, n3=12, group_col=group_col),
    ]

    for f in funcs:
        lf = f(lf)

    # 统一处理所有 feature_ 开头列的 NaN / null
    lf = lf.with_columns(
        pl.col("^feature_.*$").fill_nan(0.0).fill_null(0.0)
    )

    return lf.collect()


__all__ = [
    "add_macd",
    "add_dmi",
    "add_dma",
    "add_expma",
    "add_trix",
    "add_brar",
    "add_cr",
    "add_vr",
    "add_obv",
    "add_rsi",
    "add_wpr",
    "add_kdj",
    "add_cci",
    "add_roc",
    "add_bollinger",
    "add_turn",
    "add_bias",
    "add_mtm",
    "add_psy",
    "add_osc",
    "add_b3612",
    "add_ta",
]
