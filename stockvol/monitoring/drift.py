"""Feature + prediction drift: manual PSI (robust gauges) + Evidently HTML report.

Reference = training-period features (date <= train_cutoff).
Current   = live/post-cutoff features.

PSI is computed here directly (well-defined, no API surprises) and drives the
Prometheus gauges; Evidently produces the rich HTML drift report alongside.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index using quantile bins of the reference.

    PSI < 0.1 ~ no shift; 0.1-0.25 ~ moderate; > 0.25 ~ significant drift.
    """
    ref = reference[np.isfinite(reference)]
    cur = current[np.isfinite(current)]
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    r_hist, _ = np.histogram(ref, bins=edges)
    c_hist, _ = np.histogram(cur, bins=edges)
    eps = 1e-6
    r_pct = r_hist / max(r_hist.sum(), 1) + eps
    c_pct = c_hist / max(c_hist.sum(), 1) + eps
    return float(np.sum((c_pct - r_pct) * np.log(c_pct / r_pct)))


@dataclass
class DriftReport:
    feature_psi: dict[str, float]
    n_drifted: int
    share_drifted: float
    prediction_psi: float
    reference_dist: dict[str, float]
    current_dist: dict[str, float]
    evidently_html: str | None = None
    psi_threshold: float = 0.25

    @property
    def drifted_features(self) -> list[str]:
        return [f for f, v in self.feature_psi.items() if v > self.psi_threshold]

    def summary(self) -> str:
        top = sorted(self.feature_psi.items(), key=lambda kv: -kv[1])[:5]
        n_manual = len(self.drifted_features)
        n_feat = len(self.feature_psi)
        lines = [
            "=== Drift report (reference=train vs current=live) ===",
            # Canonical operational signal = manual PSI (drives gauges/alerts).
            f"features drifted PSI>{self.psi_threshold}: {n_manual}/{n_feat} "
            f"(share {n_manual / max(n_feat, 1):.3f})  [operational gauge]",
            f"Evidently (auto stattest): {self.n_drifted}/{n_feat} "
            f"(share {self.share_drifted:.3f})  [supplementary]",
            "top PSI: " + ", ".join(f"{f}={v:.3f}" for f, v in top),
            f"prediction-distribution PSI: {self.prediction_psi:.3f}",
            f"  train label dist: " + ", ".join(f"{k}={v:.3f}" for k, v in self.reference_dist.items()),
            f"  live  pred  dist: " + ", ".join(f"{k}={v:.3f}" for k, v in self.current_dist.items()),
        ]
        if self.evidently_html:
            lines.append(f"Evidently HTML: {self.evidently_html}")
        return "\n".join(lines)


def feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str],
    psi_threshold: float = 0.25,
) -> dict[str, float]:
    return {c: psi(reference[c].to_numpy(float), current[c].to_numpy(float)) for c in columns}


def class_distribution(values: np.ndarray, classes: list[str]) -> dict[str, float]:
    counts = np.bincount(values.astype(int), minlength=len(classes)).astype(float)
    return {classes[i]: float(counts[i] / max(counts.sum(), 1)) for i in range(len(classes))}


def run_evidently(
    reference: pd.DataFrame, current: pd.DataFrame, columns: list[str], html_path: Path
) -> tuple[int, float, str | None]:
    """Evidently DataDriftPreset -> (n_drifted, share_drifted, html_path or None)."""
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        rep = Report(metrics=[DataDriftPreset()])
        rep.run(reference_data=reference[columns], current_data=current[columns])
        res = rep.as_dict()["metrics"][0]["result"]
        html_path.parent.mkdir(parents=True, exist_ok=True)
        rep.save_html(str(html_path))
        return (
            int(res["number_of_drifted_columns"]),
            float(res["share_of_drifted_columns"]),
            str(html_path),
        )
    except Exception as exc:  # noqa: BLE001 - degrade to manual PSI only
        print(f"  (Evidently unavailable: {exc}; using manual PSI counts)")
        return -1, -1.0, None
