"""PyTorch TCN over a trailing window (HARD RULE 7).

Tensor contract
---------------
  * A sample is one (ticker, end_date) whose label is the next-day vol bucket of
    `end_date`. The input is the trailing window of `W` feature rows ending at
    `end_date` (inclusive), strictly within one ticker -> causal.
  * Input tensor:  X of shape (batch, n_features, W)  [Conv1d channels-first].
  * Target tensor: y of shape (batch,)  in {0,1,2}.

Windows are built from each ticker's contiguous processed rows (warmup already
dropped) and assigned to train/val by their END date, so a val window may look
back into train history (legitimate past) while the scaler is fit on train rows
only. Windows whose end date falls in the purge gap are discarded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .. import SEED
from ..dataset import FEATURE_COLUMNS
from ..evaluate import EvalReport, dataset_folds, score_fold
from ..scaling import CausalStandardScaler
from ..splits import Fold

WINDOW = 40
N_FEATURES = len(FEATURE_COLUMNS)


# --- model -----------------------------------------------------------------
class Chomp1d(nn.Module):
    """Trim the right padding so each conv only sees the past (causal)."""

    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp] if self.chomp > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(c_out, c_out, kernel, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """Dilated causal TCN -> last-timestep -> linear classifier (3 classes)."""

    def __init__(
        self,
        n_features: int = N_FEATURES,
        channels: tuple[int, ...] = (32, 32, 64),
        kernel: int = 3,
        dropout: float = 0.2,
        n_classes: int = 3,
    ):
        super().__init__()
        layers = []
        c_in = n_features
        for i, c_out in enumerate(channels):
            layers.append(TemporalBlock(c_in, c_out, kernel, dilation=2**i, dropout=dropout))
            c_in = c_out
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(c_in, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, F, W)
        h = self.tcn(x)          # (B, C, W)
        h = h[:, :, -1]          # last timestep summarizes the trailing window
        return self.head(h)      # (B, 3) logits


# --- windowing -------------------------------------------------------------
@dataclass
class WindowedFold:
    Xtr: np.ndarray
    ytr: np.ndarray
    Xva: np.ndarray
    yva: np.ndarray


def _ticker_windows(
    feats: np.ndarray, labels: np.ndarray, dates: np.ndarray, w: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All trailing windows of width `w` for one ticker. Returns (X, y, end_dates).

    X: (n_windows, n_features, w) ; y: (n_windows,) ; end_dates: (n_windows,).
    """
    n = len(feats)
    if n < w:
        return (np.empty((0, feats.shape[1], w)), np.empty(0), np.empty(0, dtype="datetime64[ns]"))
    idx = np.arange(w - 1, n)
    # window rows [i-w+1 .. i] -> transpose to (features, time)
    X = np.stack([feats[i - w + 1 : i + 1].T for i in idx])
    y = labels[idx]
    ed = dates[idx]
    return X, y, ed


def build_windowed_fold(
    df: pd.DataFrame, fold: Fold, w: int = WINDOW
) -> WindowedFold:
    """Scale on train rows, window per ticker, split by window END date."""
    scaler = CausalStandardScaler(columns=FEATURE_COLUMNS)
    d = pd.to_datetime(df["date"])
    train_rows = df[(d >= fold.train_start) & (d <= fold.train_end)]
    scaler.fit(train_rows)
    scaled = scaler.transform(df)

    Xtr_l, ytr_l, Xva_l, yva_l = [], [], [], []
    for _, g in scaled.groupby("ticker", sort=False):
        g = g.sort_values("date")
        feats = g[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        labels = g["label_int"].to_numpy()
        dates = pd.to_datetime(g["date"]).to_numpy()
        X, y, ed = _ticker_windows(feats, labels, dates, w)
        if len(X) == 0:
            continue
        in_train = (ed >= np.datetime64(fold.train_start)) & (ed <= np.datetime64(fold.train_end))
        in_val = (ed >= np.datetime64(fold.val_start)) & (ed <= np.datetime64(fold.val_end))
        Xtr_l.append(X[in_train]); ytr_l.append(y[in_train])
        Xva_l.append(X[in_val]); yva_l.append(y[in_val])

    return WindowedFold(
        Xtr=np.concatenate(Xtr_l), ytr=np.concatenate(ytr_l).astype(np.int64),
        Xva=np.concatenate(Xva_l), yva=np.concatenate(yva_l).astype(np.int64),
    )


# --- training --------------------------------------------------------------
@dataclass
class TrainConfig:
    max_epochs: int = 20
    patience: int = 4
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.2
    seed: int = SEED


def _loader(X: np.ndarray, y: np.ndarray, batch: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


def train_one_fold(
    wf: WindowedFold, cfg: TrainConfig, verbose: bool = False
) -> tuple[TCN, np.ndarray]:
    """Train with early stopping on val macro-F1. Returns (model, val_predictions)."""
    from sklearn.metrics import f1_score

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = TCN(dropout=cfg.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Mild class weighting (med bucket is hardest); computed from TRAIN only.
    counts = np.bincount(wf.ytr, minlength=3).astype(np.float64)
    weights = torch.tensor((counts.sum() / (3 * counts)), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    tr = _loader(wf.Xtr, wf.ytr, cfg.batch_size, shuffle=True)
    Xva = torch.from_numpy(wf.Xva)

    best_f1, best_state, best_pred, bad = -1.0, None, None, 0
    for epoch in range(cfg.max_epochs):
        model.train()
        for xb, yb in tr:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(Xva).argmax(1).numpy()
        f1 = f1_score(wf.yva, pred, labels=[0, 1, 2], average="macro", zero_division=0)
        if verbose:
            print(f"    epoch {epoch:2d}  val_macroF1={f1:.4f}")
        if f1 > best_f1 + 1e-4:
            best_f1, best_state, best_pred, bad = f1, {k: v.clone() for k, v in model.state_dict().items()}, pred, 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_pred


def run_walk_forward(
    df: pd.DataFrame, cfg: TrainConfig | None = None, w: int = WINDOW,
    verbose: bool = True, **fold_kwargs,
) -> tuple[EvalReport, list[TCN]]:
    cfg = cfg or TrainConfig()
    folds = dataset_folds(df, **fold_kwargs)
    report = EvalReport(model="TCN")
    models: list[TCN] = []
    for fold in folds:
        wf = build_windowed_fold(df, fold, w)
        if verbose:
            print(f"  fold {fold.index}: train_win={len(wf.ytr)} val_win={len(wf.yva)}")
        model, pred = train_one_fold(wf, cfg, verbose=verbose)
        report.folds.append(score_fold(fold.index, wf.yva, pred, len(wf.ytr)))
        models.append(model)
    return report, models
