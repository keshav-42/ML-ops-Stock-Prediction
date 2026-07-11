"""Persistence baseline: tomorrow's regime = today's regime. No parameters.

Volatility clusters, so the honest bar for this task is NOT the 33% random
baseline — it is persistence: bucket TODAY's GK vol against the SAME trailing
thresholds the label uses. `gk_pos` is exactly that position ((gk_t - q33)/band,
computed causally at t), so the prediction is a pure feature read:

    pos <= 0 -> low,  0 < pos <= 1 -> med,  pos > 1 -> high

Both learned models must beat this on the same walk-forward folds, otherwise
the features/model add nothing over "assume nothing changes".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluate import EvalReport, dataset_folds, score_fold
from ..splits import apply_fold


def persistence_predict(gk_pos: np.ndarray) -> np.ndarray:
    """Map today's position in the tercile band to a bucket prediction."""
    return np.where(gk_pos <= 0.0, 0, np.where(gk_pos <= 1.0, 1, 2)).astype(int)


def run_walk_forward(df: pd.DataFrame, **fold_kwargs) -> EvalReport:
    """Score persistence on the shared walk-forward folds (no training)."""
    folds = dataset_folds(df, **fold_kwargs)
    report = EvalReport(model="Persistence")
    for fold in folds:
        train, val = apply_fold(df, fold)
        pred = persistence_predict(val["gk_pos"].to_numpy())
        yva = val["label_int"].to_numpy()
        report.folds.append(score_fold(fold.index, yva, pred, len(train)))
    return report
