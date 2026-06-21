"""Phase-1 dataset assembly: raw parquet -> the (ticker, date) training table.

Per equity, within its own series (no cross-ticker bleed):
  1. price features (causal)               features.price_features
  2. labels (next-day GK vol -> tercile)   labels.make_labels
  3. join NIFTY + VIX context on EQUAL date features
  4. join deterministic calendar features  calendar_utils
  5. drop warmup rows and the final unlabeled row

VIX gaps are ffilled at most 1 trading day, and the count is logged (data_spec §3).
NIFTY context missing on an equity day -> that row is dropped (no fabrication).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .align import MARKET_FEATURES, nifty_context, vix_context
from .calendar_utils import CALENDAR_FEATURES, calendar_features
from .config import (
    EQUITY_TICKERS,
    INDEX_TICKER,
    PROCESSED_DIR,
    RAW_DIR,
    VIX_TICKER,
)
from .features import PRICE_FEATURES, price_features
from .io_utils import raw_path, read_parquet
from .labels import make_labels

FEATURE_COLUMNS = PRICE_FEATURES + MARKET_FEATURES + CALENDAR_FEATURES
# gk_t is part of FEATURE_COLUMNS; gk_next is kept for audit/closed-loop ground truth.
OUTPUT_COLUMNS = ["ticker", "date", *FEATURE_COLUMNS, "gk_next", "label", "label_int"]


@dataclass
class BuildReport:
    """Audit log of what assembly did (warmup drops, VIX fills, row counts)."""

    per_ticker: dict[str, dict] = field(default_factory=dict)
    vix_days_ffilled: int = 0
    total_rows: int = 0

    def summary(self) -> str:
        lines = [f"rows={self.total_rows}  vix_days_ffilled={self.vix_days_ffilled}"]
        for t, info in self.per_ticker.items():
            lines.append(f"  {t:14s} rows={info['rows']:5d} "
                         f"dropped_warmup={info['dropped_warmup']} "
                         f"dropped_no_ctx={info['dropped_no_ctx']}")
        return "\n".join(lines)


def _load(ticker: str, raw_dir: Path) -> pd.DataFrame:
    df = read_parquet(raw_path(raw_dir, ticker))
    if df is None or df.empty:
        raise FileNotFoundError(f"missing raw parquet for {ticker}; run ingest first")
    return df.sort_values("date").reset_index(drop=True)


def build_ticker(
    ticker: str,
    nifty_ctx: pd.DataFrame,
    vix_ctx: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    raw_dir: Path = RAW_DIR,
) -> tuple[pd.DataFrame, dict]:
    """Assemble the labeled feature table for one equity. Returns (df, stats)."""
    eq = _load(ticker, raw_dir)
    n_raw = len(eq)

    feats = price_features(eq)
    labs = make_labels(eq)

    df = pd.concat([eq[["date"]], feats], axis=1)
    # gk_t already comes from price_features; pull only the label-side columns.
    df = df.merge(labs[["gk_next", "q33", "label", "label_int"]],
                  left_index=True, right_index=True)

    # Market context joined on EQUAL date (left join keeps the equity's calendar).
    df = df.merge(nifty_ctx, on="date", how="left")
    df = df.merge(vix_ctx, on="date", how="left")

    # Calendar features (deterministic; computed against the canonical calendar).
    cal = calendar_features(pd.DatetimeIndex(df["date"]), calendar)
    df = df.merge(cal, on="date", how="left")

    df["ticker"] = ticker

    # --- row dropping (data_spec §1.3) ---
    before = len(df)
    # Warmup: any NaN in price features OR missing label threshold (q33) -> warmup.
    warmup_mask = df[PRICE_FEATURES].isna().any(axis=1) | df["q33"].isna()
    # Final unlabeled row(s): no t+1 => label_int NaN but NOT during warmup.
    no_label = df["label_int"].isna() & ~warmup_mask
    df = df[~warmup_mask].copy()
    dropped_warmup = int(warmup_mask.sum())

    # Drop rows still missing market context (no fabrication of regime).
    ctx_missing = df[["nifty_ret_1", "nifty_rstd_21", "vix_level"]].isna().any(axis=1)
    dropped_no_ctx = int(ctx_missing.sum())
    df = df[~ctx_missing].copy()

    # Drop the trailing unlabeled live row(s) from the train table.
    df = df[df["label_int"].notna()].copy()
    df["label_int"] = df["label_int"].astype(int)

    df = df[OUTPUT_COLUMNS].sort_values("date").reset_index(drop=True)
    stats = {
        "rows": len(df),
        "raw_rows": n_raw,
        "dropped_warmup": dropped_warmup,
        "dropped_no_ctx": dropped_no_ctx,
        "n_live_unlabeled": int(no_label.sum()),
    }
    return df, stats


def build_dataset(
    tickers: list[str] | None = None, raw_dir: Path = RAW_DIR
) -> tuple[pd.DataFrame, BuildReport]:
    """Build the full (ticker, date) table across all equities."""
    tickers = tickers if tickers is not None else list(EQUITY_TICKERS)

    nifty = _load(INDEX_TICKER, raw_dir)
    vix = _load(VIX_TICKER, raw_dir)
    calendar = pd.DatetimeIndex(nifty["date"]).normalize()

    nifty_ctx = nifty_context(nifty)
    vix_ctx_raw = vix_context(vix)

    # VIX: ffill <= 1 trading day on the canonical calendar; log the count.
    vix_ctx = vix_ctx_raw.set_index("date").reindex(calendar)
    filled_before = vix_ctx["vix_level"].isna().sum()
    vix_ctx = vix_ctx.ffill(limit=1)
    filled_after = vix_ctx["vix_level"].isna().sum()
    vix_days_ffilled = int(filled_before - filled_after)
    vix_ctx = vix_ctx.reset_index().rename(columns={"index": "date"})

    report = BuildReport(vix_days_ffilled=vix_days_ffilled)
    frames = []
    for t in tickers:
        df, stats = build_ticker(t, nifty_ctx, vix_ctx, calendar, raw_dir)
        frames.append(df)
        report.per_ticker[t] = stats

    full = pd.concat(frames, ignore_index=True)
    report.total_rows = len(full)
    return full, report


def write_dataset(df: pd.DataFrame, out_dir: Path = PROCESSED_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "features.parquet"
    df.to_parquet(path, engine="pyarrow", index=False)
    return path
