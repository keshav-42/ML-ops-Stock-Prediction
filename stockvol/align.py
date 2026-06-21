"""Market-regime context (NIFTY, India VIX) as date-keyed feature frames.

These are computed on each series' OWN timeline (causal), then joined onto an
equity by EQUAL date `t` (never t+1) in dataset.py. data_spec §2/§3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MARKET_FEATURES = ["nifty_ret_1", "nifty_rstd_21", "vix_level", "vix_chg"]


def nifty_context(nifty: pd.DataFrame) -> pd.DataFrame:
    """nifty_ret_1, nifty_rstd_21 on the NIFTY close. Returns date-keyed frame."""
    df = nifty.sort_values("date").reset_index(drop=True)
    c = df["close"]
    r = np.log(c / c.shift(1))
    out = pd.DataFrame({"date": df["date"]})
    out["nifty_ret_1"] = r
    out["nifty_rstd_21"] = r.rolling(21).std()
    return out


def vix_context(vix: pd.DataFrame) -> pd.DataFrame:
    """vix_level (close at t) and vix_chg (t - t-1). Returns date-keyed frame."""
    df = vix.sort_values("date").reset_index(drop=True)
    out = pd.DataFrame({"date": df["date"]})
    out["vix_level"] = df["close"]
    out["vix_chg"] = df["close"].diff()
    return out
