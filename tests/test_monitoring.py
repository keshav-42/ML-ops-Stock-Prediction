"""Phase-5 monitoring tests (offline)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockvol.monitoring.drift import class_distribution, feature_drift, psi


def test_psi_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    b = rng.normal(0, 1, 5000)
    assert psi(a, b) < 0.05  # same distribution -> ~0


def test_psi_large_for_shifted_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    b = rng.normal(2.0, 1, 5000)  # big mean shift
    assert psi(a, b) > 0.25  # significant drift


def test_feature_drift_flags_shifted_column():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 2000), "y": rng.normal(0, 1, 2000)})
    cur = pd.DataFrame({"x": rng.normal(0, 1, 2000), "y": rng.normal(3, 1, 2000)})
    d = feature_drift(ref, cur, ["x", "y"])
    assert d["x"] < 0.1
    assert d["y"] > 0.25


def test_class_distribution_sums_to_one():
    vals = np.array([0, 0, 1, 2, 2, 2])
    dist = class_distribution(vals, ["low", "med", "high"])
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    assert dist["high"] == 0.5
