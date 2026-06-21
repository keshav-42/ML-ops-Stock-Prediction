"""Closed-loop monitoring — the demo centerpiece.

Replays the genuinely out-of-sample period (dates after the model's train cutoff):
for each (ticker, date) the served model predicts the next-day vol bucket, and the
ACTUAL next-day bucket — the `label_int` already in the processed table, computed
via trailing terciles — is the ground truth. We track rolling live accuracy and
macro-F1 and raise a decay alert when accuracy falls below a threshold.

This is the honest "did the model actually work in production?" loop, run on data
the model never saw during training.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from ..labels import INT_TO_LABEL
from ..serving.inference import Predictor


def predict_period(predictor: Predictor, start_date: str) -> pd.DataFrame:
    """Batch-predict every (ticker, date) with date >= start_date that has a label.

    Returns long frame: ticker, date, pred_int, actual_int.
    """
    w = predictor.window
    mean, std = predictor._mean, predictor._std
    cols = predictor.columns
    start = pd.Timestamp(start_date)
    out = []

    for ticker, g in predictor._store.items():
        g = g.reset_index(drop=True)
        feats = g[cols].to_numpy(dtype=np.float32)
        labels = g["label_int"].to_numpy()
        dates = g["date"].to_numpy()
        # valid window end positions: full lookback AND date >= start
        ends = [i for i in range(w - 1, len(g)) if dates[i] >= np.datetime64(start)]
        if not ends:
            continue
        X = np.stack([((feats[i - w + 1 : i + 1] - mean) / std).T for i in ends]).astype(np.float32)
        with torch.no_grad():
            preds = predictor.model(torch.from_numpy(X)).argmax(1).numpy()
        for j, i in enumerate(ends):
            out.append((ticker, pd.Timestamp(dates[i]), int(preds[j]), int(labels[i])))

    return pd.DataFrame(out, columns=["ticker", "date", "pred_int", "actual_int"])


@dataclass
class ClosedLoopReport:
    n_predictions: int
    overall_accuracy: float
    overall_macro_f1: float
    recall_per_class: dict
    rolling: pd.DataFrame  # date, n, daily_acc, rolling_acc, rolling_f1
    decay_alert: bool
    decay_threshold: float
    min_rolling_acc: float
    eval_start: str
    eval_end: str

    def summary(self) -> str:
        alert = "  *** DECAY ALERT ***" if self.decay_alert else ""
        rc = "/".join(f"{self.recall_per_class[k]:.3f}" for k in ("low", "med", "high"))
        return (
            f"=== Closed-loop live accuracy ({self.eval_start} .. {self.eval_end}) ===\n"
            f"out-of-sample predictions: {self.n_predictions}\n"
            f"overall accuracy: {self.overall_accuracy:.4f} | macro-F1: {self.overall_macro_f1:.4f}"
            f" | recall(l/m/h) {rc}\n"
            f"rolling-accuracy min: {self.min_rolling_acc:.4f} "
            f"(threshold {self.decay_threshold:.2f}){alert}"
        )


def run_closed_loop(
    predictor: Predictor,
    start_date: str | None = None,
    rolling_days: int = 21,
    decay_threshold: float = 0.40,
) -> ClosedLoopReport:
    """Replay the post-cutoff period and compute rolling live metrics."""
    start_date = start_date or predictor.train_cutoff
    df = predict_period(predictor, start_date)
    df = df.sort_values("date").reset_index(drop=True)

    y, p = df["actual_int"].to_numpy(), df["pred_int"].to_numpy()
    acc = float((y == p).mean())
    macro = float(f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0))
    from sklearn.metrics import recall_score

    rec = recall_score(y, p, labels=[0, 1, 2], average=None, zero_division=0)
    recall = {INT_TO_LABEL[i]: float(rec[i]) for i in range(3)}

    # Per-day aggregates, then a rolling window over days.
    daily = (
        df.assign(correct=(df["pred_int"] == df["actual_int"]).astype(int))
        .groupby("date")
        .agg(n=("correct", "size"), daily_acc=("correct", "mean"))
        .reset_index()
    )
    daily["rolling_acc"] = daily["daily_acc"].rolling(rolling_days, min_periods=5).mean()

    # rolling macro-F1 over the trailing `rolling_days` of raw predictions
    roll_f1 = []
    dvals = daily["date"].to_numpy()
    for d in dvals:
        lo = pd.Timestamp(d) - pd.Timedelta(days=rolling_days * 2)
        m = (df["date"] <= d) & (df["date"] > lo)
        sub = df[m]
        roll_f1.append(
            f1_score(sub["actual_int"], sub["pred_int"], labels=[0, 1, 2],
                     average="macro", zero_division=0) if len(sub) > 20 else np.nan
        )
    daily["rolling_f1"] = roll_f1

    min_roll = float(np.nanmin(daily["rolling_acc"].to_numpy())) if len(daily) else acc
    decay = bool(min_roll < decay_threshold)

    return ClosedLoopReport(
        n_predictions=len(df),
        overall_accuracy=acc,
        overall_macro_f1=macro,
        recall_per_class=recall,
        rolling=daily,
        decay_alert=decay,
        decay_threshold=decay_threshold,
        min_rolling_acc=min_roll,
        eval_start=str(df["date"].min().date()),
        eval_end=str(df["date"].max().date()),
    )
