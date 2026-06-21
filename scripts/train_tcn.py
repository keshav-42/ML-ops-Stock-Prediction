"""Train + evaluate the PyTorch TCN on the SAME walk-forward folds; log to MLflow.

    python -m scripts.train_tcn
    python -m scripts.train_tcn --max-epochs 15 --folds 5
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import pandas as pd
import torch

from stockvol.config import PROCESSED_DIR
from stockvol.models.tcn import WINDOW, TrainConfig, run_walk_forward

try:
    import mlflow

    _HAVE_MLFLOW = True
except ImportError:  # pragma: no cover
    _HAVE_MLFLOW = False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--threads", type=int, default=None)
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    cfg = TrainConfig(max_epochs=args.max_epochs, patience=args.patience)
    report, models = run_walk_forward(df, cfg, n_folds=args.folds)
    print("\n" + report.summary())

    metrics = {
        "macro_f1_mean": report.mean_macro_f1,
        "macro_f1_std": report.std_macro_f1,
        **{f"recall_{k}": v for k, v in report.mean_recall().items()},
    }
    out = PROCESSED_DIR / "tcn_metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

    # Compare against the baseline if present.
    base_path = PROCESSED_DIR / "baseline_metrics.json"
    if base_path.exists():
        base = json.loads(base_path.read_text())
        delta = report.mean_macro_f1 - base["macro_f1_mean"]
        verdict = "BEATS" if delta > 0 else "does NOT beat"
        print(f"\nTCN {verdict} baseline: "
              f"{report.mean_macro_f1:.4f} vs {base['macro_f1_mean']:.4f} "
              f"(Δ={delta:+.4f})")

    if _HAVE_MLFLOW:
        mlflow.set_experiment("volatility-bucket")
        with mlflow.start_run(run_name="tcn"):
            mlflow.log_params({**asdict(cfg), "window": WINDOW})
            mlflow.log_metrics(metrics)
            for f in report.folds:
                mlflow.log_metric("fold_macro_f1", f.macro_f1, step=f.fold)
            mlflow.log_artifact(str(out))
            torch.save(models[-1].state_dict(), PROCESSED_DIR / "tcn_lastfold.pt")
            mlflow.log_artifact(str(PROCESSED_DIR / "tcn_lastfold.pt"))
        print("logged to MLflow (./mlruns)")


if __name__ == "__main__":
    main()
