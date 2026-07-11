"""Causal feature engineering (data_spec.md §2).

Every function takes a SINGLE ticker's OHLCV frame, sorted ascending by date, and
returns features where row `t` reads only days `<= t`. No `center=True`, no
future-filled NaNs, EMAs with `adjust=False`. Rolling windows are trailing.

These are deliberately small pure functions so the leakage tests can exercise the
exact same code paths the dataset build uses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LN2 = np.log(2.0)
GK_C2 = 2.0 * LN2 - 1.0  # ≈ 0.386294 (Garman–Klass open->close coefficient)

# Trailing tercile-threshold window (shared with labels.py — the label is bucketed
# against these same thresholds, so features may read them causally at t).
TRAILING_WINDOW = 252
Q_LOW = 1.0 / 3.0
Q_HIGH = 2.0 / 3.0

# Feature columns produced from a single equity series (excludes market+calendar).
PRICE_FEATURES = [
    "ret_1", "ret_5", "ret_10", "ret_21",
    "rstd_5", "rstd_10", "rstd_21",
    "parkinson_t", "gk_t", "gk_ma_5", "gk_ma_10", "gk_ma_21",
    "atr_14", "rsi_14", "macd", "macd_signal", "macd_hist", "vol_z_21",
    "gk_pos", "gk_ma5_pos", "gk_edge_dist", "gk_vov_21", "gk_trend",
]


def gk_vol(df: pd.DataFrame) -> pd.Series:
    """Single-day Garman–Klass volatility (data_spec §1.1). Reads O,H,L,C of day t."""
    hl = np.log(df["high"] / df["low"])
    co = np.log(df["close"] / df["open"])
    var = 0.5 * hl**2 - GK_C2 * co**2
    return np.sqrt(var.clip(lower=0.0))  # floor before sqrt; never NaN from neg var


def trailing_thresholds(gk: pd.Series, window: int = TRAILING_WINDOW) -> pd.DataFrame:
    """q33/q66 of GK vol over a trailing window ENDING at t (includes t, not t+1).

    `min_periods=window` => the first `window-1` rows have NaN thresholds and are
    dropped downstream (insufficient history). Lives here (not labels.py) because
    the threshold geometry is BOTH the label's bucketing rule and a legitimate
    causal feature: at day t the thresholds are fully known.
    """
    q33 = gk.rolling(window, min_periods=window).quantile(Q_LOW)
    q66 = gk.rolling(window, min_periods=window).quantile(Q_HIGH)
    return pd.DataFrame({"q33": q33, "q66": q66})


def parkinson_vol(df: pd.DataFrame) -> pd.Series:
    """Parkinson range volatility. Reads H,L of day t."""
    hl = np.log(df["high"] / df["low"])
    return np.sqrt((1.0 / (4.0 * LN2)) * hl**2)


def _wilder_ema(s: pd.Series, n: int) -> pd.Series:
    """Wilder smoothing == EMA with alpha=1/n, adjust=False (causal)."""
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def atr_norm(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder ATR(n) normalized by close (scale-free). Reads H,L,C_t and C_{t-1}."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return _wilder_ema(tr, n) / df["close"]


def rsi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder RSI(n) on close. Reads close through t."""
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = _wilder_ema(gain, n)
    avg_loss = _wilder_ema(loss, n)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # All-gain / all-loss edge cases.
    out = out.where(avg_loss != 0.0, 100.0)
    out = out.where(avg_gain != 0.0, out.where(avg_loss == 0.0, 0.0))
    return out


def macd(df: pd.DataFrame) -> pd.DataFrame:
    """MACD(12,26,9) with causal EMAs (adjust=False). Reads close through t."""
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    return pd.DataFrame({"macd": line, "macd_signal": signal, "macd_hist": line - signal})


def price_features(df: pd.DataFrame) -> pd.DataFrame:
    """All single-ticker price features. `df` must be one ticker, sorted by date.

    Returns a frame indexed like `df` with `PRICE_FEATURES` columns.
    """
    c = df["close"]
    r = np.log(c / c.shift(1))  # daily log return r_t

    out = pd.DataFrame(index=df.index)
    for k in (1, 5, 10, 21):
        out[f"ret_{k}"] = np.log(c / c.shift(k))
    for k in (5, 10, 21):
        out[f"rstd_{k}"] = r.rolling(k).std()

    out["parkinson_t"] = parkinson_vol(df)
    gk = gk_vol(df)
    out["gk_t"] = gk
    for k in (5, 10, 21):
        out[f"gk_ma_{k}"] = gk.rolling(k).mean()

    out["atr_14"] = atr_norm(df, 14)
    out["rsi_14"] = rsi(df, 14)
    out = pd.concat([out, macd(df)], axis=1)

    v = df["volume"]
    vmean = v.rolling(21).mean()
    vstd = v.rolling(21).std()
    out["vol_z_21"] = (v - vmean) / vstd.replace(0.0, np.nan)

    # Label-geometry features: the label buckets GK_vol_{t+1} against trailing
    # tercile thresholds known at t — give the model today's position in that
    # exact geometry (vol persistence makes this the strongest signal).
    thr = trailing_thresholds(gk)
    band = (thr["q66"] - thr["q33"]).replace(0.0, np.nan)
    out["gk_pos"] = (gk - thr["q33"]) / band            # <=0 low band, <=1 med, >1 high
    out["gk_ma5_pos"] = (out["gk_ma_5"] - thr["q33"]) / band
    out["gk_edge_dist"] = (
        np.minimum((gk - thr["q33"]).abs(), (gk - thr["q66"]).abs()) / band
    )  # small => near a boundary => regime flip likely

    # Vol-of-vol (dispersion of recent vol) and short/long vol regime slope.
    out["gk_vov_21"] = gk.rolling(21).std() / gk.rolling(21).mean().replace(0.0, np.nan)
    out["gk_trend"] = out["gk_ma_5"] / out["gk_ma_21"].replace(0.0, np.nan) - 1.0

    return out
