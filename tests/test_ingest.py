"""Phase-0 ingestion tests — offline (no network).

Cover the contracts that matter for a re-runnable pipeline: schema normalization
(incl. the MultiIndex / no-Adj-Close cases) and idempotent merge/dedup.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockvol.config import IngestConfig
from stockvol.ingest import merge_idempotent, normalize_ohlcv
from stockvol.io_utils import raw_path, read_parquet, ticker_to_filename, write_parquet


def _yf_like_frame(dates: list[str], multiindex: bool, ticker: str) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(dates), name="Date")
    n = len(dates)
    data = {
        "Open": np.linspace(100, 110, n),
        "High": np.linspace(101, 112, n),
        "Low": np.linspace(99, 108, n),
        "Close": np.linspace(100, 111, n),
        "Volume": np.arange(n, dtype=float) * 1000 + 5000,
    }
    df = pd.DataFrame(data, index=idx)
    if multiindex:  # newer yfinance: (field, ticker) columns, no Adj Close
        df.columns = pd.MultiIndex.from_product([df.columns, [ticker]])
    return df


def test_normalize_flat_columns():
    raw = _yf_like_frame(["2020-01-01", "2020-01-02"], multiindex=False, ticker="X.NS")
    out = normalize_ohlcv(raw, "X.NS", auto_adjust=True)
    assert list(out.columns) == [
        "date", "open", "high", "low", "close", "volume", "adj_close",
        "ticker", "asset_class",
    ]
    assert out["ticker"].unique().tolist() == ["X.NS"]
    assert out["asset_class"].unique().tolist() == ["equity"]
    # auto_adjust -> adj_close mirrors close
    assert (out["adj_close"] == out["close"]).all()
    # tz-naive normalized dates
    assert out["date"].dt.tz is None


def test_normalize_multiindex_columns():
    raw = _yf_like_frame(["2020-01-01", "2020-01-02"], multiindex=True, ticker="^NSEI")
    out = normalize_ohlcv(raw, "^NSEI", auto_adjust=True)
    assert out["asset_class"].unique().tolist() == ["index"]
    assert len(out) == 2


def test_merge_idempotent_dedups_on_date():
    a = normalize_ohlcv(
        _yf_like_frame(["2020-01-01", "2020-01-02"], False, "X.NS"), "X.NS", True
    )
    b = normalize_ohlcv(
        _yf_like_frame(["2020-01-02", "2020-01-03"], False, "X.NS"), "X.NS", True
    )
    merged = merge_idempotent(a, b)
    # union of 3 unique dates, sorted, no duplicate 2020-01-02
    assert merged["date"].dt.date.astype(str).tolist() == [
        "2020-01-01", "2020-01-02", "2020-01-03",
    ]
    assert merged["date"].is_monotonic_increasing


def test_merge_keeps_latest_fetch_on_conflict():
    a = normalize_ohlcv(_yf_like_frame(["2020-01-02"], False, "X.NS"), "X.NS", True)
    b = a.copy()
    b.loc[:, "close"] = 999.0  # newer fetch corrects the value
    merged = merge_idempotent(a, b)
    assert len(merged) == 1
    assert merged["close"].iloc[0] == 999.0


def test_write_then_read_roundtrip_and_rerun_noop(tmp_path):
    cfg = IngestConfig(raw_dir=tmp_path)
    df = normalize_ohlcv(
        _yf_like_frame(["2020-01-01", "2020-01-02"], False, "X.NS"), "X.NS", True
    )
    path = raw_path(cfg.raw_dir, "X.NS")
    write_parquet(df, path)

    # Simulate a second run merging the same data: row count must not grow.
    existing = read_parquet(path)
    merged = merge_idempotent(existing, df)
    write_parquet(merged, path)
    assert len(read_parquet(path)) == 2


def test_manifest_merge_does_not_clobber(tmp_path):
    """A subset refresh must update only its entries, not wipe the rest."""
    from stockvol.ingest import _read_manifest, _write_manifest

    _write_manifest(tmp_path, {"A.NS": {"ticker": "A.NS", "rows": 10}})
    # Simulate ingest_universe's merge step for a different ticker.
    manifest = _read_manifest(tmp_path)
    manifest["B.NS"] = {"ticker": "B.NS", "rows": 20}
    _write_manifest(tmp_path, manifest)

    final = _read_manifest(tmp_path)
    assert set(final) == {"A.NS", "B.NS"}
    assert final["A.NS"]["rows"] == 10


def test_ticker_to_filename():
    assert ticker_to_filename("RELIANCE.NS") == "RELIANCE_NS"
    assert ticker_to_filename("^NSEI") == "_NSEI"
