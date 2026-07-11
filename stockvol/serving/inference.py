"""Predictor: builds the trailing window, scales it, runs the quantized model.

Loads the artifact (FP32 weights + scaler + meta), static-quantizes at startup
using the saved calibration sample, and loads the processed feature table as the
serving feature store. Trailing windows end at the as-of date and read only days
<= t (causal), consistent with training.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..config import PROCESSED_DIR
from ..models.tcn import TCN
from ..quantize import static_quantize
from .artifact import ARTIFACT_DIR, load_meta, load_scaler


class InsufficientHistory(Exception):
    pass


class UnknownTicker(Exception):
    pass


class Predictor:
    def __init__(
        self,
        artifact_dir: Path = ARTIFACT_DIR,
        feature_path: Path | None = None,
        quantize: bool | None = None,
    ):
        self.meta = load_meta(artifact_dir)
        sc = load_scaler(artifact_dir)
        self.columns: list[str] = self.meta["feature_columns"]
        self.window: int = self.meta["window"]
        self.classes: list[str] = self.meta["classes"]
        self.train_cutoff: str = self.meta["train_cutoff"]
        self._mean = np.asarray(sc["mean"], dtype=np.float32)
        self._std = np.asarray(sc["std"], dtype=np.float32)

        # n_features from the artifact, not the code — the artifact pins its own
        # feature list, which may differ from the current FEATURE_COLUMNS.
        model = TCN(n_features=len(self.columns), channels=tuple(self.meta["channels"]))
        model.load_state_dict(torch.load(artifact_dir / "model_fp32.pt"))
        model.eval()

        if quantize is None:
            quantize = os.environ.get("SERVE_QUANTIZE", "1") == "1"
        if quantize:
            calib = np.load(artifact_dir / "calib.npy")
            self.model = static_quantize(model, calib)
            self.model_quant = "static-INT8"
        else:
            self.model = model
            self.model_quant = "FP32"

        # Feature store + canonical calendar for next-day lookup.
        fp = feature_path or (PROCESSED_DIR / "features.parquet")
        store = pd.read_parquet(fp)
        store["date"] = pd.to_datetime(store["date"])
        self._store = {t: g.sort_values("date").reset_index(drop=True)
                       for t, g in store.groupby("ticker", sort=False)}
        self._calendar = np.array(sorted(store["date"].unique()), dtype="datetime64[ns]")

    @property
    def tickers(self) -> list[str]:
        return list(self._store)

    def _next_trading_day(self, date: np.datetime64) -> str:
        pos = self._calendar.searchsorted(date, side="right")
        if pos < len(self._calendar):
            return str(pd.Timestamp(self._calendar[pos]).date())
        # past the last known day -> estimate next business day
        return str((pd.Timestamp(date) + pd.offsets.BDay(1)).date())

    def replay(self, ticker: str, days: int = 60) -> list[dict]:
        """Predicted vs actual buckets for the last `days` labeled trading days.

        Same causal batch replay as monitoring.closed_loop, restricted to one
        ticker — backs the dashboard's hit/miss ribbon. Only rows whose t+1
        label is already known are included, so this never grades the pending
        (still-unknown) next-day prediction.
        """
        if ticker not in self._store:
            raise UnknownTicker(ticker)
        g = self._store[ticker]
        labeled = g.index[(g.index >= self.window - 1) & g["label_int"].notna()]
        ends = list(labeled[-days:])
        if not ends:
            return []

        feats = g[self.columns].to_numpy(dtype=np.float32)
        X = np.stack(
            [((feats[i - self.window + 1 : i + 1] - self._mean) / self._std).T for i in ends]
        ).astype(np.float32)
        with torch.no_grad():
            preds = self.model(torch.from_numpy(X)).argmax(1).numpy()

        return [
            {
                "date": str(pd.Timestamp(g.iloc[i]["date"]).date()),
                "predicted": self.classes[int(preds[j])],
                "actual": self.classes[int(g.iloc[i]["label_int"])],
            }
            for j, i in enumerate(ends)
        ]

    def _scaled_window(self, ticker: str, date: str | None) -> tuple[pd.DataFrame, int, np.ndarray]:
        """Resolve the as-of row and return (group, end_idx, scaled window)."""
        if ticker not in self._store:
            raise UnknownTicker(ticker)
        g = self._store[ticker]

        if date is None:
            end_idx = len(g) - 1
        else:
            d = pd.Timestamp(date)
            matches = g.index[g["date"] <= d]
            if len(matches) == 0:
                raise InsufficientHistory(f"no data on/before {date} for {ticker}")
            end_idx = int(matches[-1])

        if end_idx + 1 < self.window:
            raise InsufficientHistory(
                f"need {self.window} rows, have {end_idx + 1} for {ticker}"
            )

        win = g.iloc[end_idx - self.window + 1 : end_idx + 1]
        feats = win[self.columns].to_numpy(dtype=np.float32)
        scaled = (feats - self._mean) / self._std
        return g, end_idx, scaled

    def predict(self, ticker: str, date: str | None = None) -> dict:
        g, end_idx, scaled = self._scaled_window(ticker, date)
        x = torch.from_numpy(scaled.T[None, :, :])  # (1, n_features, window)

        with torch.no_grad():
            probs = F.softmax(self.model(x), dim=1).numpy()[0]
        bucket = self.classes[int(probs.argmax())]
        as_of = pd.Timestamp(g.iloc[end_idx]["date"])

        return {
            "ticker": ticker,
            "as_of_date": str(as_of.date()),
            "predicted_for": self._next_trading_day(np.datetime64(as_of)),
            "bucket": bucket,
            "probs": {c: float(p) for c, p in zip(self.classes, probs)},
            "model_quant": self.model_quant,
        }

    def explain(self, ticker: str, date: str | None = None) -> dict:
        """Occlusion attribution for the current prediction.

        Each feature channel is zeroed in the standardized window (0 == the
        train-fold mean, i.e. 'this feature is unremarkable'); the drop in the
        predicted class's probability is that feature's contribution. Positive
        means the feature pushed the model TOWARD the predicted bucket.
        Gradient-free, so it works on the INT8 static-quantized model.
        """
        g, end_idx, scaled = self._scaled_window(ticker, date)
        n = len(self.columns)

        base = torch.from_numpy(scaled.T[None, :, :])
        occluded = np.repeat(scaled.T[None, :, :], n, axis=0)  # (n, F, W)
        for j in range(n):
            occluded[j, j, :] = 0.0

        with torch.no_grad():
            probs = F.softmax(self.model(base), dim=1).numpy()[0]
            k = int(probs.argmax())
            abl = F.softmax(
                self.model(torch.from_numpy(occluded.astype(np.float32))), dim=1
            ).numpy()[:, k]

        raw_last = g.iloc[end_idx]
        attributions = sorted(
            (
                {
                    "feature": c,
                    "contribution": float(probs[k] - abl[j]),
                    "value": float(raw_last[c]),
                }
                for j, c in enumerate(self.columns)
            ),
            key=lambda a: abs(a["contribution"]),
            reverse=True,
        )
        return {
            "ticker": ticker,
            "as_of_date": str(pd.Timestamp(raw_last["date"]).date()),
            "bucket": self.classes[k],
            "prob": float(probs[k]),
            "attributions": attributions,
        }
