"""Time-series splits: expanding-window walk-forward with a purge gap.

HARD RULE 2/5: never shuffle; leave a purge gap (>= the 1-day label horizon)
between train-end and val-start so the last train label (computed from t+1)
cannot fall inside the validation feature window.

Splits are defined on the GLOBAL date axis and applied to every ticker, so a fold
boundary is the same calendar date for all tickers (no cross-ticker leakage and
no windows straddling the boundary).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp  # inclusive
    val_start: pd.Timestamp  # inclusive (after purge gap)
    val_end: pd.Timestamp    # inclusive


def make_walk_forward_folds(
    dates: pd.Series | pd.DatetimeIndex,
    n_folds: int = 5,
    purge_gap: int = 1,
    min_train: int = 252,
) -> list[Fold]:
    """Expanding-window folds over the sorted unique trading dates.

    Each fold trains on [start .. train_end], purges `purge_gap` trading days, then
    validates on the next contiguous block. Validation blocks are equal-sized and
    non-overlapping; training expands each fold.
    """
    uniq = pd.DatetimeIndex(pd.Series(pd.to_datetime(dates)).dropna().unique()).sort_values()
    n = len(uniq)
    if n < min_train + n_folds * (purge_gap + 1):
        raise ValueError("not enough dates for the requested folds/purge")

    val_total = n - min_train - purge_gap
    val_size = val_total // n_folds
    folds: list[Fold] = []
    for k in range(n_folds):
        train_end_idx = min_train - 1 + k * val_size
        val_start_idx = train_end_idx + 1 + purge_gap
        val_end_idx = val_start_idx + val_size - 1 if k < n_folds - 1 else n - 1
        if val_start_idx > val_end_idx:
            break
        folds.append(
            Fold(
                index=k,
                train_start=uniq[0],
                train_end=uniq[train_end_idx],
                val_start=uniq[val_start_idx],
                val_end=uniq[val_end_idx],
            )
        )
    return folds


def apply_fold(df: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a (ticker, date) table into (train, val) by the fold's date bounds."""
    d = pd.to_datetime(df["date"])
    train = df[(d >= fold.train_start) & (d <= fold.train_end)]
    val = df[(d >= fold.val_start) & (d <= fold.val_end)]
    return train.copy(), val.copy()
