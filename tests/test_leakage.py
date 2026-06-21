"""LEAKAGE SUITE — the correctness gate to Phase 2 (data_spec.md §4, HARD RULE 5).

These tests are the analog of unit tests for the math: they assert that no feature
peeks into the future, that labels use only trailing thresholds, that splits don't
overlap, that scaling is train-only, and that no feature is suspiciously perfectly
correlated with the label. A full pass is the stop-the-line gate to model code.

Each test maps to a numbered invariant in data_spec §4 / the Rule-5 checklist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockvol.align import nifty_context, vix_context
from stockvol.calendar_utils import CALENDAR_FEATURES, calendar_features
from stockvol.dataset import FEATURE_COLUMNS, build_ticker, build_dataset
from stockvol.features import PRICE_FEATURES, price_features
from stockvol.labels import TRAILING_WINDOW, make_labels, trailing_thresholds
from stockvol.scaling import CausalStandardScaler
from stockvol.splits import apply_fold, make_walk_forward_folds

MARKET_DERIVED = [c for c in FEATURE_COLUMNS if c not in CALENDAR_FEATURES]


# --- synthetic fixtures (deterministic, no network) ------------------------
def _synth_ohlcv(n: int = 600, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    ret = rng.normal(0, 0.012, n)
    close = 100 * np.exp(np.cumsum(ret))
    # Build a plausible OHLC around close with positive ranges.
    spread = np.abs(rng.normal(0, 0.008, n)) * close
    high = close + spread
    low = close - spread
    open_ = low + rng.random(n) * (high - low)
    vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"date": dates, "open": open_, "high": high, "low": low,
         "close": close, "volume": vol}
    )


@pytest.fixture(scope="module")
def ohlcv() -> pd.DataFrame:
    return _synth_ohlcv()


# --- §4.1 / Rule 5: NO FUTURE BLEED ---------------------------------------
def test_future_row_does_not_change_past_features(ohlcv):
    """Mutating day t+1 (and beyond) must not change ANY feature at day t.

    This is the strongest causal probe: it holds even for EMA-based features
    (ATR/RSI/MACD) which depend on all history but only PAST history.
    """
    base = price_features(ohlcv)

    tampered = ohlcv.copy()
    last = len(tampered) - 1
    # Drastically corrupt the final bar (a "future" relative to every prior row).
    tampered.loc[last, ["open", "high", "low", "close"]] *= 5.0
    tampered.loc[last, "volume"] *= 100.0
    after = price_features(tampered)

    # All rows except the tampered last one must be bit-identical.
    pd.testing.assert_frame_equal(base.iloc[:-1], after.iloc[:-1])


def test_truncation_invariance(ohlcv):
    """Computing features on data[:t+1] equals computing on full then slicing.

    data_spec §4.1 stated directly. Checked at several interior points.
    """
    full = price_features(ohlcv)
    for t in (260, 400, 599):
        trunc = price_features(ohlcv.iloc[: t + 1])
        for col in PRICE_FEATURES:
            a = full[col].iloc[t]
            b = trunc[col].iloc[t]
            if pd.isna(a) and pd.isna(b):
                continue
            assert np.isclose(a, b, rtol=1e-9, atol=1e-12), f"{col} bled at t={t}"


def test_shift_by_one_day_shifts_features_identically(ohlcv):
    """Rule 5: delaying the whole series by one bar shifts features identically.

    Insert a duplicate of the first bar at the front; every original bar now sits
    one position later and its (post-warmup) feature values must be unchanged.
    """
    base = price_features(ohlcv).reset_index(drop=True)
    shifted_in = pd.concat([ohlcv.iloc[[0]], ohlcv], ignore_index=True)
    shifted = price_features(shifted_in).reset_index(drop=True)
    # Compare well past warmup so fixed-window features are fully populated.
    for col in ["ret_1", "ret_5", "rstd_21", "gk_t", "parkinson_t", "vol_z_21"]:
        a = base[col].iloc[300]
        b = shifted[col].iloc[301]  # same underlying bar, one position later
        assert np.isclose(a, b, rtol=1e-9, atol=1e-12), f"{col} not shift-equivariant"


# --- §4.2 / §1: ONLY t+1 QUANTITY IS THE LABEL ----------------------------
def test_label_uses_next_day_and_trailing_thresholds(ohlcv):
    """label_t buckets GK_vol_{t+1} against thresholds from days <= t."""
    labs = make_labels(ohlcv)
    gk = labs["gk_t"]
    thr = trailing_thresholds(gk)
    # gk_next is exactly gk shifted up by one.
    assert np.allclose(
        labs["gk_next"].iloc[:-1].to_numpy(),
        gk.shift(-1).iloc[:-1].to_numpy(),
        equal_nan=True,
    )
    # Reconstruct a label at an interior valid point and check it matches.
    t = 400
    q33, q66 = thr["q33"].iloc[t], thr["q66"].iloc[t]
    gnext = labs["gk_next"].iloc[t]
    expected = 0 if gnext <= q33 else (1 if gnext <= q66 else 2)
    assert int(labs["label_int"].iloc[t]) == expected


def test_thresholds_are_trailing_only(ohlcv):
    """§4.3: q33/q66 at t use only GK vol on days <= t (truncation-invariant)."""
    gk = make_labels(ohlcv)["gk_t"]
    full = trailing_thresholds(gk)
    for t in (300, 500):
        trunc = trailing_thresholds(gk.iloc[: t + 1])
        assert np.isclose(full["q33"].iloc[t], trunc["q33"].iloc[t])
        assert np.isclose(full["q66"].iloc[t], trunc["q66"].iloc[t])


def test_threshold_window_excludes_t_plus_1(ohlcv):
    """A spike on day t+1 must not move the thresholds dated at day t."""
    gk = make_labels(ohlcv)["gk_t"].copy()
    t = 400
    before = trailing_thresholds(gk)["q66"].iloc[t]
    gk.iloc[t + 1] = gk.max() * 10  # corrupt the future
    after = trailing_thresholds(gk)["q66"].iloc[t]
    assert np.isclose(before, after)


# --- §4.5 / Rule 5: CLEAN SPLITS ------------------------------------------
def test_split_boundaries_do_not_overlap():
    dates = pd.bdate_range("2015-01-01", periods=1500)
    folds = make_walk_forward_folds(dates, n_folds=5, purge_gap=1, min_train=252)
    assert len(folds) == 5
    for f in folds:
        # purge gap: at least one trading day strictly between train_end and val_start
        gap = (dates.searchsorted(f.val_start) - dates.searchsorted(f.train_end))
        assert gap >= 2, f"fold {f.index} purge gap too small ({gap})"
        assert f.train_end < f.val_start
    # Validation blocks are non-overlapping and ordered.
    for a, b in zip(folds, folds[1:]):
        assert a.val_end < b.val_start


def test_no_window_straddles_train_val_boundary():
    """Applying a fold yields train rows strictly <= train_end < val rows."""
    df = pd.DataFrame({"date": pd.bdate_range("2015-01-01", periods=1000),
                       "x": range(1000)})
    folds = make_walk_forward_folds(df["date"], n_folds=3)
    tr, va = apply_fold(df, folds[0])
    assert tr["date"].max() <= folds[0].train_end < va["date"].min()


# --- §4.4 / Rule 5: CAUSAL SCALING ----------------------------------------
def test_scaler_params_come_only_from_train(ohlcv):
    cols = ["gk_t", "ret_1", "vix_level"]
    feats = price_features(ohlcv)
    feats["vix_level"] = np.linspace(10, 30, len(feats))  # stand-in column
    df = feats.dropna().reset_index(drop=True)
    split = len(df) // 2
    train, val = df.iloc[:split], df.iloc[split:]

    sc = CausalStandardScaler(columns=cols).fit(train)
    # Fitting on train+val must NOT reproduce the train-only params (proves val
    # rows did not influence the fitted statistics).
    sc_all = CausalStandardScaler(columns=cols).fit(df)
    assert not np.allclose(sc.params["mean"], sc_all.params["mean"])
    # Recomputing the params by hand from train only matches.
    assert np.allclose(sc.params["mean"], train[cols].mean().to_numpy())


# --- §4.6: NO CROSS-TICKER BLEED ------------------------------------------
def test_no_cross_ticker_bleed():
    """A ticker built in isolation == its slice from the full multi-ticker build."""
    full, _ = build_dataset()
    one, _ = build_dataset(tickers=["RELIANCE.NS"])
    sub = full[full["ticker"] == "RELIANCE.NS"].reset_index(drop=True)
    one = one.reset_index(drop=True)
    assert len(sub) == len(one)
    for col in FEATURE_COLUMNS + ["label_int"]:
        np.testing.assert_allclose(
            sub[col].to_numpy(dtype=float), one[col].to_numpy(dtype=float),
            rtol=1e-9, atol=1e-12, err_msg=f"{col} differs across build scope",
        )


# --- §4.7: WARMUP DROPPED, NO NaNs ----------------------------------------
def test_no_nans_and_warmup_dropped():
    full, report = build_dataset(tickers=["RELIANCE.NS"])
    assert full[FEATURE_COLUMNS].isna().to_numpy().sum() == 0
    assert full["label_int"].isna().sum() == 0
    # First usable row must be at/after the 252-row threshold warmup.
    raw_first = report.per_ticker["RELIANCE.NS"]["raw_rows"]
    assert report.per_ticker["RELIANCE.NS"]["dropped_warmup"] >= TRAILING_WINDOW - 1
    assert full["label_int"].isin([0, 1, 2]).all()


# --- §4.8: SANITY CORRELATION ---------------------------------------------
def test_no_feature_near_perfectly_correlated_with_label():
    full, _ = build_dataset()
    y = full["label_int"].to_numpy()
    for col in FEATURE_COLUMNS:
        x = full[col].to_numpy(dtype=float)
        if np.std(x) == 0:
            continue
        r = abs(np.corrcoef(x, y)[0, 1])
        assert r < 0.6, f"{col} suspiciously correlated with label (|r|={r:.3f})"


# --- calendar features: deterministic, schedule known in advance ----------
def test_calendar_features_depend_only_on_date(ohlcv):
    """Calendar block is a pure function of date + the (forward) schedule, not of
    prices. Given two calendars that BOTH fully cover the target dates, every
    calendar feature is identical regardless of which OHLCV produced the dates.

    (The expiry features legitimately need the forward schedule to be known — they
    encode a published-in-advance F&O calendar, available at day t. The only place
    a too-short calendar matters is an incomplete final month, which the real build
    never hits because it uses the full NIFTY calendar.)
    """
    cal = pd.DatetimeIndex(ohlcv["date"])  # ~600 business days
    cal_a = pd.bdate_range("2015-01-01", periods=900)   # fully covers `cal`
    cal_b = pd.bdate_range("2015-01-01", periods=1100)  # also fully covers `cal`
    a = calendar_features(cal, cal_a)
    b = calendar_features(cal, cal_b)
    for col in CALENDAR_FEATURES:
        np.testing.assert_allclose(a[col].to_numpy(), b[col].to_numpy(),
                                   err_msg=f"{col} not schedule-deterministic")


def test_expiry_is_last_thursday_rolled_back():
    from stockvol.calendar_utils import monthly_expiries

    cal = pd.bdate_range("2020-01-01", "2020-12-31")  # Mon-Fri business days
    exps = monthly_expiries(cal)
    # Jan 2020: last Thursday is 2020-01-30.
    jan = [e for e in exps if e.month == 1 and e.year == 2020][0]
    assert jan == pd.Timestamp("2020-01-30")
    assert jan.weekday() == 3
