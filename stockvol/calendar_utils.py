"""NSE trading-calendar helpers and India-specific F&O expiry features.

The trading calendar is taken from the realized NSE sessions (we use the NIFTY
index's traded dates as the canonical calendar). The monthly F&O expiry is the
LAST THURSDAY of the month, rolled back to the prior trading day on a holiday.

Leakage note: every quantity here is a deterministic function of the *date* and
the published-in-advance trading/holiday + expiry schedule. None of it reads
prices/volume/VIX, so it carries no market-data lookahead. `days_to_expiry`
counts forward to a *scheduled* date that is knowable at day `t`. These features
are therefore exempt from the market-feature truncation-invariance test and are
checked separately (same date -> same value regardless of surrounding prices).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CALENDAR_FEATURES = [
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "expiry_week_flag",
    "days_to_expiry",
]


def monthly_expiries(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last-Thursday monthly expiries, each rolled back to a real trading day.

    For every (year, month) spanned by `calendar`, take the last Thursday; if that
    Thursday is not a trading day (holiday), roll back to the latest trading day on
    or before it. Returns sorted, de-duplicated expiry dates.
    """
    cal = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(calendar).normalize())))
    trading = set(cal)
    expiries: list[pd.Timestamp] = []

    periods = pd.period_range(cal.min(), cal.max(), freq="M")
    for per in periods:
        # All Thursdays in this month.
        days = pd.date_range(per.start_time, per.end_time, freq="D")
        thursdays = days[days.weekday == 3]
        if len(thursdays) == 0:
            continue
        exp = thursdays[-1]
        # Roll back to the nearest trading day <= last Thursday.
        while exp not in trading and exp >= cal.min():
            exp -= pd.Timedelta(days=1)
        if exp in trading:
            expiries.append(exp)

    return pd.DatetimeIndex(sorted(set(expiries)))


def calendar_features(dates: pd.DatetimeIndex, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute the deterministic calendar feature block for `dates`.

    `calendar` is the canonical trading calendar (>= the span of `dates`), used to
    derive expiries and to count trading days to the next expiry.
    """
    dates = pd.DatetimeIndex(dates).normalize()
    cal = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(calendar).normalize())))
    expiries = monthly_expiries(cal)

    # Cyclical encodings (period 7 for weekday, 12 for month).
    wd = dates.weekday.to_numpy()
    mo = dates.month.to_numpy()
    out = pd.DataFrame(index=dates)
    out["dow_sin"] = np.sin(2 * np.pi * wd / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * wd / 7.0)
    out["month_sin"] = np.sin(2 * np.pi * (mo - 1) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * (mo - 1) / 12.0)

    # Map each trading day to its position in the calendar, and each expiry too,
    # so "days to next expiry" is a count of TRADING days (not calendar days).
    cal_pos = {d: i for i, d in enumerate(cal)}
    expiry_positions = np.array([cal_pos[e] for e in expiries if e in cal_pos])
    expiry_iso = {(e.isocalendar().year, e.isocalendar().week) for e in expiries}

    days_to_expiry = np.full(len(dates), np.nan)
    expiry_week_flag = np.zeros(len(dates), dtype=int)
    for i, d in enumerate(dates):
        iso = d.isocalendar()
        if (iso.year, iso.week) in expiry_iso:
            expiry_week_flag[i] = 1
        pos = cal_pos.get(d)
        if pos is None or len(expiry_positions) == 0:
            continue
        # Next expiry at or after this trading day.
        future = expiry_positions[expiry_positions >= pos]
        if len(future) > 0:
            days_to_expiry[i] = int(future[0] - pos)

    out["expiry_week_flag"] = expiry_week_flag
    out["days_to_expiry"] = days_to_expiry
    return out.reset_index().rename(columns={"index": "date"})
