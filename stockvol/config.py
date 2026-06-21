"""Project configuration and the ticker universe.

Typed with pydantic so config is validated and self-documenting (HARD RULE 9).
Paths are resolved relative to the repo root so the package works from any CWD.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field

# Repo root = parent of the `stockvol` package directory.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = REPO_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"

# --- Ticker universe -------------------------------------------------------
# 25 liquid NIFTY-50 constituents. yfinance uses the `.NS` (NSE) suffix.
EQUITY_TICKERS: tuple[str, ...] = (
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "HINDUNILVR.NS",
    "ITC.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "AXISBANK.NS",
    "BAJFINANCE.NS",
    "ASIANPAINT.NS",
    "MARUTI.NS",
    "HCLTECH.NS",
    "SUNPHARMA.NS",
    "TITAN.NS",
    "ULTRACEMCO.NS",
    "WIPRO.NS",
    "NESTLEIND.NS",
    "M&M.NS",  # Mahindra & Mahindra — replaces TATAMOTORS.NS (2025 demerger broke .NS symbol)
    "TATASTEEL.NS",
    "POWERGRID.NS",
    "NTPC.NS",
)

# NIFTY-50 index (market-regime context) and India VIX (first-class vol feature).
INDEX_TICKER: str = "^NSEI"
VIX_TICKER: str = "^INDIAVIX"

# asset_class tag stored alongside each series, so downstream code can treat
# equities / index / vix differently without re-deriving from the symbol.
ASSET_CLASS: dict[str, str] = {
    INDEX_TICKER: "index",
    VIX_TICKER: "vix",
}


def all_tickers() -> tuple[str, ...]:
    """Full download universe: equities + index + VIX."""
    return (*EQUITY_TICKERS, INDEX_TICKER, VIX_TICKER)


def asset_class_of(ticker: str) -> str:
    """Return 'equity' | 'index' | 'vix' for a ticker."""
    return ASSET_CLASS.get(ticker, "equity")


class IngestConfig(BaseModel):
    """Configuration for Phase-0 ingestion."""

    start: date = Field(default=date(2012, 1, 1), description="Inclusive history start.")
    end: date | None = Field(default=None, description="Exclusive end; None = today.")
    raw_dir: Path = Field(default=RAW_DIR)
    # auto_adjust=True -> yfinance returns split/dividend-adjusted O/H/L/C, which is
    # required for consistent range-based volatility (data_spec assumption).
    auto_adjust: bool = Field(default=True)
    max_retries: int = Field(default=3, ge=1)
    retry_sleep_s: float = Field(default=2.0, ge=0)
    # On re-run, re-fetch this many days before the stored last_date to absorb any
    # late vendor corrections, then merge+dedup. Keeps ingest idempotent and current.
    refetch_buffer_days: int = Field(default=7, ge=0)

    model_config = {"arbitrary_types_allowed": True}
