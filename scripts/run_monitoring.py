"""Run the full model-monitoring pass: closed-loop live accuracy + drift.

    python -m scripts.run_monitoring

Writes:
  data/monitoring/closed_loop.csv     per-day rolling accuracy/F1
  data/monitoring/monitoring.json     summary metrics
  data/monitoring/drift_report.html   Evidently report
  data/monitoring/metrics.prom        Prometheus textfile (+ optional pushgateway)
"""

from __future__ import annotations

import json
import math

import pandas as pd

from stockvol.config import PROCESSED_DIR, REPO_ROOT
from stockvol.dataset import FEATURE_COLUMNS
from stockvol.monitoring.closed_loop import predict_period, run_closed_loop
from stockvol.monitoring.drift import (
    DriftReport,
    class_distribution,
    feature_drift,
    run_evidently,
)
from stockvol.monitoring.metrics_export import export
from stockvol.serving.inference import Predictor

MON_DIR = REPO_ROOT / "data" / "monitoring"


def main() -> None:
    predictor = Predictor()
    cutoff = pd.Timestamp(predictor.train_cutoff)

    # --- closed loop (centerpiece) ---
    cl = run_closed_loop(predictor)
    print(cl.summary())
    MON_DIR.mkdir(parents=True, exist_ok=True)
    cl.rolling.to_csv(MON_DIR / "closed_loop.csv", index=False)

    # --- drift: reference=train period, current=live period ---
    store = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    store["date"] = pd.to_datetime(store["date"])
    ref = store[store["date"] <= cutoff]
    cur = store[store["date"] > cutoff]

    feat_psi = feature_drift(ref, cur, FEATURE_COLUMNS)
    n_drift, share, html = run_evidently(
        ref, cur, FEATURE_COLUMNS, MON_DIR / "drift_report.html"
    )
    if n_drift < 0:  # Evidently degraded -> derive from manual PSI
        drifted = [f for f, v in feat_psi.items() if v > 0.25]
        n_drift, share = len(drifted), len(drifted) / len(feat_psi)

    # prediction-distribution drift: train labels vs live predictions
    classes = predictor.classes
    ref_dist = class_distribution(ref["label_int"].to_numpy(), classes)
    live = predict_period(predictor, predictor.train_cutoff)
    cur_dist = class_distribution(live["pred_int"].to_numpy(), classes)
    pred_psi = sum(
        ((cur_dist[c] + 1e-6) - (ref_dist[c] + 1e-6))
        * math.log((cur_dist[c] + 1e-6) / (ref_dist[c] + 1e-6))
        for c in classes
    )

    drift = DriftReport(
        feature_psi=feat_psi,
        n_drifted=n_drift,
        share_drifted=share,
        prediction_psi=float(pred_psi),
        reference_dist=ref_dist,
        current_dist=cur_dist,
        evidently_html=html,
    )
    print("\n" + drift.summary())

    # --- persist + export prometheus metrics ---
    export(cl, drift, MON_DIR / "metrics.prom")
    summary = {
        "closed_loop": {
            "n_predictions": cl.n_predictions,
            "overall_accuracy": cl.overall_accuracy,
            "overall_macro_f1": cl.overall_macro_f1,
            "recall_per_class": cl.recall_per_class,
            "min_rolling_acc": cl.min_rolling_acc,
            "decay_alert": cl.decay_alert,
            "eval_start": cl.eval_start,
            "eval_end": cl.eval_end,
        },
        "drift": {
            "n_drifted": drift.n_drifted,
            "share_drifted": drift.share_drifted,
            "drifted_features": drift.drifted_features,
            "prediction_psi": drift.prediction_psi,
            "reference_dist": ref_dist,
            "current_dist": cur_dist,
        },
    }
    (MON_DIR / "monitoring.json").write_text(json.dumps(summary, indent=2), "utf-8")
    print(f"\nwrote {MON_DIR / 'monitoring.json'} and metrics.prom")


if __name__ == "__main__":
    main()
