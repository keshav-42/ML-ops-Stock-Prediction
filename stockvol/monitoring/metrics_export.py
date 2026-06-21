"""Export monitoring gauges for Prometheus.

Writes a textfile (node_exporter textfile-collector format) and optionally pushes
to a Pushgateway (PUSHGATEWAY_URL). Keeps the batch monitoring jobs decoupled from
the always-on API while still surfacing in the same Prometheus/Grafana stack.
"""

from __future__ import annotations

import os
from pathlib import Path

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway, write_to_textfile

from .closed_loop import ClosedLoopReport
from .drift import DriftReport


def build_registry(cl: ClosedLoopReport, drift: DriftReport) -> CollectorRegistry:
    reg = CollectorRegistry()

    g_acc = Gauge("live_accuracy", "Out-of-sample rolling-period accuracy", registry=reg)
    g_f1 = Gauge("live_macro_f1", "Out-of-sample macro-F1", registry=reg)
    g_minroll = Gauge("live_min_rolling_accuracy", "Min rolling accuracy", registry=reg)
    g_decay = Gauge("model_decay_alert", "1 if rolling accuracy < threshold", registry=reg)
    g_npred = Gauge("live_predictions_total", "Closed-loop predictions evaluated", registry=reg)
    g_acc.set(cl.overall_accuracy)
    g_f1.set(cl.overall_macro_f1)
    g_minroll.set(cl.min_rolling_acc)
    g_decay.set(1 if cl.decay_alert else 0)
    g_npred.set(cl.n_predictions)

    g_share = Gauge("feature_drift_share", "Share of features drifted (PSI)", registry=reg)
    g_ndrift = Gauge("feature_drift_count", "Count of drifted features", registry=reg)
    g_pred_psi = Gauge("prediction_drift_psi", "Prediction-distribution PSI", registry=reg)
    g_share.set(cl_share(drift))
    g_ndrift.set(len(drift.drifted_features))
    g_pred_psi.set(drift.prediction_psi)

    g_feat = Gauge("feature_psi", "Per-feature PSI", ["feature"], registry=reg)
    for feat, val in drift.feature_psi.items():
        g_feat.labels(feature=feat).set(val)

    return reg


def cl_share(drift: DriftReport) -> float:
    return len(drift.drifted_features) / max(len(drift.feature_psi), 1)


def export(cl: ClosedLoopReport, drift: DriftReport, textfile: Path) -> None:
    reg = build_registry(cl, drift)
    textfile.parent.mkdir(parents=True, exist_ok=True)
    write_to_textfile(str(textfile), reg)
    gateway = os.environ.get("PUSHGATEWAY_URL")
    if gateway:
        try:
            push_to_gateway(gateway, job="volatility_monitoring", registry=reg)
            print(f"  pushed metrics to {gateway}")
        except Exception as exc:  # noqa: BLE001
            print(f"  (pushgateway {gateway} unavailable: {exc})")
