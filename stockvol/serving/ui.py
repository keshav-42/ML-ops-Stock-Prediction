"""Read-only endpoints that back the Next.js dashboard (web/).

/tickers   — the serving universe (from the loaded feature store).
/history   — raw OHLCV candles for one ticker (query param: `&` in M&M.NS
             breaks path params, so ticker is always a query arg here).
/ribbon    — predicted-vs-actual buckets for the last N labeled days
             (single-ticker closed-loop replay via Predictor.replay).
/accuracy  — rolling live accuracy/F1 series from data/monitoring/closed_loop.csv.
/explain   — occlusion feature attribution for the current prediction.

All endpoints are read-only; only /ribbon and /explain touch the model.
"""

from __future__ import annotations

import math

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from ..config import DATA_DIR, RAW_DIR
from ..io_utils import raw_path
from .inference import InsufficientHistory, Predictor, UnknownTicker

router = APIRouter()

CLOSED_LOOP_CSV = DATA_DIR / "monitoring" / "closed_loop.csv"


def _predictor() -> Predictor:
    # Lazy import: app.py includes this router, so a top-level import would be circular.
    from .app import STATE

    pred = STATE.get("predictor")
    if pred is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    return pred


@router.get("/tickers")
def tickers() -> dict:
    pred = _predictor()
    return {"tickers": sorted(pred.tickers), "window": pred.window,
            "train_cutoff": pred.train_cutoff}


@router.get("/history")
def history(
    ticker: str = Query(..., examples=["RELIANCE.NS"]),
    days: int = Query(default=180, ge=10, le=2000),
) -> dict:
    path = raw_path(RAW_DIR, ticker)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"unknown ticker: {ticker}")
    df = pd.read_parquet(path).sort_values("date").tail(days)
    candles = [
        {
            "date": str(pd.Timestamp(r.date).date()),
            "open": round(float(r.open), 2),
            "high": round(float(r.high), 2),
            "low": round(float(r.low), 2),
            "close": round(float(r.close), 2),
            "volume": int(r.volume),
        }
        for r in df.itertuples()
    ]
    return {"ticker": ticker, "candles": candles}


@router.get("/ribbon")
def ribbon(
    ticker: str = Query(..., examples=["RELIANCE.NS"]),
    days: int = Query(default=60, ge=5, le=250),
) -> dict:
    try:
        entries = _predictor().replay(ticker, days)
    except UnknownTicker as e:
        raise HTTPException(status_code=404, detail=f"unknown ticker: {e}") from e
    hits = sum(e["predicted"] == e["actual"] for e in entries)
    return {
        "ticker": ticker,
        "entries": entries,
        "hit_rate": round(hits / len(entries), 4) if entries else None,
    }


@router.get("/explain")
def explain(
    ticker: str = Query(..., examples=["RELIANCE.NS"]),
    date: str | None = Query(default=None, description="As-of date YYYY-MM-DD."),
) -> dict:
    try:
        return _predictor().explain(ticker, date)
    except UnknownTicker as e:
        raise HTTPException(status_code=404, detail=f"unknown ticker: {e}") from e
    except InsufficientHistory as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.get("/accuracy")
def accuracy(days: int = Query(default=120, ge=10, le=1000)) -> dict:
    if not CLOSED_LOOP_CSV.exists():
        raise HTTPException(status_code=404, detail="closed-loop metrics not generated yet")
    df = pd.read_csv(CLOSED_LOOP_CSV).tail(days)

    def _clean(v: float) -> float | None:
        return None if (v is None or (isinstance(v, float) and math.isnan(v))) else round(v, 4)

    series = [
        {
            "date": str(r.date),
            "daily_acc": _clean(r.daily_acc),
            "rolling_acc": _clean(r.rolling_acc),
            "rolling_f1": _clean(r.rolling_f1),
        }
        for r in df.itertuples()
    ]
    latest = series[-1] if series else None
    return {"series": series, "latest": latest}
