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

        model = TCN(channels=tuple(self.meta["channels"]))
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

    def predict(self, ticker: str, date: str | None = None) -> dict:
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
