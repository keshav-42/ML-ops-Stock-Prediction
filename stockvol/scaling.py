"""Causal standardization (HARD RULE 3).

A StandardScaler-style transformer whose statistics are fit on the TRAIN fold only
and then applied to val/test. Kept tiny and explicit so the leakage suite can
assert the fitted params depend on no val/test row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CausalStandardScaler:
    """Fit mean/std on train rows only; transform any rows with those stats."""

    columns: list[str]
    mean_: pd.Series | None = None
    std_: pd.Series | None = None

    def fit(self, train: pd.DataFrame) -> "CausalStandardScaler":
        self.mean_ = train[self.columns].mean()
        # ddof=0 for a population estimate; guard zero-variance columns.
        std = train[self.columns].std(ddof=0)
        self.std_ = std.replace(0.0, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("scaler not fitted")
        out = df.copy()
        out[self.columns] = (df[self.columns] - self.mean_) / self.std_
        return out

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train).transform(train)

    @property
    def params(self) -> dict[str, np.ndarray]:
        assert self.mean_ is not None and self.std_ is not None
        return {"mean": self.mean_.to_numpy(), "std": self.std_.to_numpy()}
