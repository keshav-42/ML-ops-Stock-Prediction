"""Label construction (data_spec.md §1) — next-day GK vol -> trailing tercile.

The ONLY place a `t+1` quantity enters the dataset. Thresholds use a trailing
window ending at `t` (does NOT include t+1); the value being bucketed is the
realized GK vol on `t+1`. This is the leakage-free supervised setup.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# trailing_thresholds lives in features.py (the same geometry is a causal feature);
# re-exported here so label code and tests keep importing it from labels.
from .features import (  # noqa: F401
    Q_HIGH,
    Q_LOW,
    TRAILING_WINDOW,
    gk_vol,
    trailing_thresholds,
)

LABELS = ("low", "med", "high")
LABEL_TO_INT = {"low": 0, "med": 1, "high": 2}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


def make_labels(df: pd.DataFrame, window: int = TRAILING_WINDOW) -> pd.DataFrame:
    """Build label columns for one ticker's OHLCV frame (sorted by date).

    Returns columns: gk_t, gk_next, q33, q66, label (str), label_int.
    `label` is NaN on the final row (no t+1) and during warmup (no full window).
    """
    gk = gk_vol(df)
    thr = trailing_thresholds(gk, window)
    gk_next = gk.shift(-1)  # GK_vol_{t+1} bucketed against thresholds known at t

    out = pd.DataFrame(index=df.index)
    out["gk_t"] = gk
    out["gk_next"] = gk_next
    out["q33"] = thr["q33"]
    out["q66"] = thr["q66"]

    label_int = np.full(len(df), np.nan)
    valid = thr["q33"].notna() & gk_next.notna()
    gn = gk_next.to_numpy()
    q33 = thr["q33"].to_numpy()
    q66 = thr["q66"].to_numpy()
    v = valid.to_numpy()
    # low if gk_next <= q33; med if q33 < gk_next <= q66; high if > q66.
    label_int[v & (gn <= q33)] = 0
    label_int[v & (gn > q33) & (gn <= q66)] = 1
    label_int[v & (gn > q66)] = 2

    out["label_int"] = label_int
    out["label"] = pd.Series(label_int, index=df.index).map(INT_TO_LABEL)
    return out
