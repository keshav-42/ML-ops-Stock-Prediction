"""Train + evaluate the LightGBM baseline; log to MLflow.

    python -m scripts.train_baseline
"""

from __future__ import annotations

import json

import pandas as pd

from stockvol.config import PROCESSED_DIR
from stockvol.models.lgbm_baseline import DEFAULT_PARAMS, run_walk_forward
from stockvol.models.persistence import run_walk_forward as run_persistence

try:
    import mlflow

    _HAVE_MLFLOW = True
except ImportError:  # pragma: no cover
    _HAVE_MLFLOW = False


def main() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")

    # Persistence first: the no-model bar every learned model must beat.
    persist = run_persistence(df)
    print(persist.summary())
    persist_metrics = {
        "macro_f1_mean": persist.mean_macro_f1,
        "macro_f1_std": persist.std_macro_f1,
        **{f"recall_{k}": v for k, v in persist.mean_recall().items()},
    }
    pout = PROCESSED_DIR / "persistence_metrics.json"
    pout.write_text(json.dumps(persist_metrics, indent=2), encoding="utf-8")
    print(f"wrote {pout}\n")

    report, _ = run_walk_forward(df)
    print(report.summary())

    metrics = {
        "macro_f1_mean": report.mean_macro_f1,
        "macro_f1_std": report.std_macro_f1,
        **{f"recall_{k}": v for k, v in report.mean_recall().items()},
    }
    out = PROCESSED_DIR / "baseline_metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

    if _HAVE_MLFLOW:
        mlflow.set_experiment("volatility-bucket")
        with mlflow.start_run(run_name="lgbm-baseline"):
            mlflow.log_params(DEFAULT_PARAMS)
            mlflow.log_metrics(metrics)
            for f in report.folds:
                mlflow.log_metric("fold_macro_f1", f.macro_f1, step=f.fold)
            mlflow.log_artifact(str(out))
        print("logged to MLflow (./mlruns)")


if __name__ == "__main__":
    main()
