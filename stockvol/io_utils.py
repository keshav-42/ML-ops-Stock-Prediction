"""Parquet read/write helpers with near-atomic writes.

Writing to a temp file then replacing avoids leaving a half-written parquet if the
process dies mid-write — important because ingest is meant to be re-runnable.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def ticker_to_filename(ticker: str) -> str:
    """`RELIANCE.NS` -> `RELIANCE_NS`; `^NSEI` -> `_NSEI`. Filesystem-safe stem."""
    return ticker.replace(".", "_").replace("^", "_")


def raw_path(raw_dir: Path, ticker: str) -> Path:
    return raw_dir / f"{ticker_to_filename(ticker)}.parquet"


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write `df` to `path` atomically (tmp file + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, engine="pyarrow", index=False)
    os.replace(tmp, path)  # atomic on the same filesystem


def read_parquet(path: Path) -> pd.DataFrame | None:
    """Read parquet, or return None if the file does not exist."""
    if not path.exists():
        return None
    return pd.read_parquet(path, engine="pyarrow")
