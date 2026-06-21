"""LightGBM baseline (HARD RULE 6) — the honest gate the TCN must beat.

Pooled multi-ticker, walk-forward over the shared folds. Trees don't need scaling,
so features are used as-is. Reports macro-F1 + per-class recall on the same splits
the TCN will use. Seeded for reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from .. import SEED
from ..dataset import FEATURE_COLUMNS
from ..evaluate import EvalReport, dataset_folds, score_fold
from ..splits import apply_fold

DEFAULT_PARAMS = dict(
    objective="multiclass",
    num_class=3,
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    min_child_samples=50,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)


def run_walk_forward(
    df: pd.DataFrame, params: dict | None = None, **fold_kwargs
) -> tuple[EvalReport, list]:
    """Train+eval a fresh LightGBM per fold. Returns (report, fitted_models)."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    folds = dataset_folds(df, **fold_kwargs)
    report = EvalReport(model="LightGBM")
    models = []

    for fold in folds:
        train, val = apply_fold(df, fold)
        Xtr, ytr = train[FEATURE_COLUMNS].to_numpy(), train["label_int"].to_numpy()
        Xva, yva = val[FEATURE_COLUMNS].to_numpy(), val["label_int"].to_numpy()

        clf = LGBMClassifier(**params)
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xva)

        report.folds.append(score_fold(fold.index, yva, np.asarray(pred), len(ytr)))
        models.append(clf)

    return report, models
