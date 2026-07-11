"""Market-regime context (NIFTY, India VIX) as date-keyed feature frames.

These are computed on each series' OWN timeline (causal), then joined onto an
equity by EQUAL date `t` (never t+1) in dataset.py. data_spec §2/§3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import gk_vol

MARKET_FEATURES = [
    "nifty_ret_1", "nifty_rstd_21", "nifty_gk",
    "vix_level", "vix_chg", "vix_z_63",
]


def nifty_context(nifty: pd.DataFrame) -> pd.DataFrame:
    """nifty_ret_1, nifty_rstd_21, nifty_gk on the NIFTY series. Date-keyed frame."""
    df = nifty.sort_values("date").reset_index(drop=True)
    c = df["close"]
    r = np.log(c / c.shift(1))
    out = pd.DataFrame({"date": df["date"]})
    out["nifty_ret_1"] = r
    out["nifty_rstd_21"] = r.rolling(21).std()
    out["nifty_gk"] = gk_vol(df)  # market-wide range vol at t
    return out


def vix_context(vix: pd.DataFrame) -> pd.DataFrame:
    """vix_level (close at t), vix_chg (t - t-1), vix_z_63 (level vs trailing 63d).

    The z-score makes the nonstationary VIX level usable across regimes.
    Returns a date-keyed frame.
    """
    df = vix.sort_values("date").reset_index(drop=True)
    out = pd.DataFrame({"date": df["date"]})
    out["vix_level"] = df["close"]
    out["vix_chg"] = df["close"].diff()
    m = df["close"].rolling(63).mean()
    s = df["close"].rolling(63).std()
    out["vix_z_63"] = (df["close"] - m) / s.replace(0.0, np.nan)
    return out
