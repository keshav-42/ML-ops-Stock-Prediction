"""Phase 0 — idempotent daily-OHLCV ingestion from yfinance to parquet.

Design notes (see architecture.md §2):
  * One parquet file per ticker under `data/raw/`.
  * `auto_adjust=True` -> split/dividend-adjusted O/H/L/C (consistent range vol).
  * Re-running merges new dates into the existing file and dedups on `date`,
    keeping the most recently fetched row. Running twice is a no-op on data.
  * We store exactly what the exchange returned per ticker. We do NOT forward-fill
    missing trading days here — alignment/cleaning is a Phase-1 concern so that the
    raw store never fabricates bars (data_spec §3).
"""

from __future__ import annotations

import json
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

from .config import IngestConfig, all_tickers, asset_class_of
from .io_utils import raw_path, read_parquet, write_parquet

CANONICAL_COLS = ["date", "open", "high", "low", "close", "volume", "adj_close"]
MANIFEST_NAME = "_manifest.json"


# --- normalization ---------------------------------------------------------
def normalize_ohlcv(raw: pd.DataFrame, ticker: str, auto_adjust: bool) -> pd.DataFrame:
    """Coerce a yfinance frame into the canonical schema.

    Handles both flat columns and the MultiIndex (price, ticker) layout newer
    yfinance versions emit, and the presence/absence of `Adj Close`.
    """
    df = raw.copy()

    # Flatten a possible (field, ticker) MultiIndex down to the field level.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()  # Date index -> column
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Date column may be named 'date' or 'datetime' depending on interval.
    date_col = "date" if "date" in df.columns else "datetime"
    df = df.rename(columns={date_col: "date"})

    # With auto_adjust=True there is no 'adj_close'; close is already adjusted.
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df[CANONICAL_COLS]
    df["ticker"] = ticker
    df["asset_class"] = asset_class_of(ticker)

    # Drop rows with no usable price; coerce numerics.
    for col in ("open", "high", "low", "close", "adj_close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def merge_idempotent(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Union of existing+new rows, deduped on `date` keeping the latest fetch."""
    if existing is None or existing.empty:
        combined = new
    else:
        combined = pd.concat([existing, new], ignore_index=True)
    # `keep="last"` -> the freshly fetched row wins over a stale stored one.
    combined = combined.drop_duplicates(subset=["date"], keep="last")
    return combined.sort_values("date").reset_index(drop=True)


# --- fetching --------------------------------------------------------------
def fetch_one(
    ticker: str, start: date, end: date | None, cfg: IngestConfig
) -> pd.DataFrame:
    """Download daily OHLCV for one ticker with retries. Raises on repeated failure."""
    import yfinance as yf

    last_err: Exception | None = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            raw = yf.download(
                ticker,
                start=start.isoformat(),
                end=None if end is None else end.isoformat(),
                interval="1d",
                auto_adjust=cfg.auto_adjust,
                actions=False,
                progress=False,
                threads=False,
            )
            if raw is not None and not raw.empty:
                return normalize_ohlcv(raw, ticker, cfg.auto_adjust)
            last_err = RuntimeError("empty frame returned")
        except Exception as exc:  # noqa: BLE001 - retry any transient vendor error
            last_err = exc
        if attempt < cfg.max_retries:
            time.sleep(cfg.retry_sleep_s * attempt)
    raise RuntimeError(f"failed to fetch {ticker} after {cfg.max_retries} tries: {last_err}")


def _resume_start(existing: pd.DataFrame | None, cfg: IngestConfig) -> date:
    """Where to start fetching: config start, or just before the stored last date."""
    if existing is None or existing.empty:
        return cfg.start
    last = pd.to_datetime(existing["date"]).max().date()
    resumed = last - timedelta(days=cfg.refetch_buffer_days)
    return max(resumed, cfg.start)


def ingest_ticker(ticker: str, cfg: IngestConfig) -> dict:
    """Fetch (resume-aware), merge, and persist one ticker. Returns manifest entry."""
    path = raw_path(cfg.raw_dir, ticker)
    existing = read_parquet(path)
    start = _resume_start(existing, cfg)

    new = fetch_one(ticker, start, cfg.end, cfg)
    merged = merge_idempotent(existing, new)
    write_parquet(merged, path)

    return {
        "ticker": ticker,
        "asset_class": asset_class_of(ticker),
        "rows": int(len(merged)),
        "first_date": str(merged["date"].min().date()),
        "last_date": str(merged["date"].max().date()),
        "fetched_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "path": str(path),
    }


def ingest_universe(
    cfg: IngestConfig | None = None, tickers: list[str] | None = None
) -> dict[str, dict]:
    """Ingest the full universe (or a subset). Continues past per-ticker failures."""
    cfg = cfg or IngestConfig()
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    symbols = tickers if tickers is not None else list(all_tickers())

    # Start from the existing manifest so a subset re-run updates only its own
    # entries instead of clobbering the rest (nightly partial-refresh friendly).
    manifest = _read_manifest(cfg.raw_dir)
    for sym in symbols:
        try:
            entry = ingest_ticker(sym, cfg)
            manifest[sym] = entry
            print(f"  [ok]   {sym:14s} {entry['rows']:5d} rows "
                  f"[{entry['first_date']} .. {entry['last_date']}]")
        except Exception as exc:  # noqa: BLE001 - record + keep going
            manifest[sym] = {"ticker": sym, "error": str(exc)}
            print(f"  [FAIL] {sym:14s} {exc}")

    _write_manifest(cfg.raw_dir, manifest)
    return manifest


def _read_manifest(raw_dir: Path) -> dict[str, dict]:
    path = raw_dir / MANIFEST_NAME
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_manifest(raw_dir: Path, manifest: dict[str, dict]) -> None:
    path = raw_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
