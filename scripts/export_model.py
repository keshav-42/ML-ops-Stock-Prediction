"""Train the pooled production TCN on all history and export the serving artifact.

    python -m scripts.export_model

Trains on rows up to a time cutoff (with a small held-out tail for early stopping),
fits the causal scaler on the train rows, and saves FP32 weights + scaler + meta +
a calibration sample (used to static-quantize at serving startup).
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from stockvol.config import PROCESSED_DIR
from stockvol.dataset import FEATURE_COLUMNS
from stockvol.labels import LABELS
from stockvol.models.tcn import WINDOW, TCN, TrainConfig, build_windowed_fold
from stockvol.scaling import CausalStandardScaler
from stockvol.serving.artifact import ProductionArtifact, save_artifact
from stockvol.transfer import _fit, single_temporal_fold


def main() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    # Expanding split: train on the bulk, hold out a recent tail for early stopping.
    fold = single_temporal_fold(df, train_frac=0.9)
    wf = build_windowed_fold(df, fold)

    cfg = TrainConfig(max_epochs=25, patience=5)
    print(f"training pooled production TCN: train_win={len(wf.ytr)} val_win={len(wf.yva)}")
    model = _fit(TCN(dropout=cfg.dropout), wf, cfg, trainable="all")
    model.eval()

    # Scaler fit on the same train rows (causal).
    scaler = CausalStandardScaler(columns=FEATURE_COLUMNS).fit(
        df[pd.to_datetime(df["date"]) <= fold.train_end]
    )

    # Calibration sample for startup static quantization.
    rng = np.random.default_rng(42)
    idx = rng.choice(len(wf.Xtr), size=min(512, len(wf.Xtr)), replace=False)
    calib = wf.Xtr[idx]

    artifact = ProductionArtifact(
        feature_columns=list(FEATURE_COLUMNS),
        classes=list(LABELS),
        window=WINDOW,
        channels=[32, 32, 64],
        train_cutoff=str(fold.train_end.date()),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        scaler_mean=scaler.mean_.tolist(),
        scaler_std=scaler.std_.tolist(),
    )
    out = save_artifact(artifact, model.state_dict(), calib)
    print(f"saved artifact -> {out}")
    print(f"  train_cutoff={artifact.train_cutoff}  classes={artifact.classes}")


if __name__ == "__main__":
    main()
