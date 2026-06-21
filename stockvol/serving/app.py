"""FastAPI app: /predict, /health, /metrics (Prometheus).

The quantized model + feature store load ONCE at startup (lifespan). /predict
checks Redis first (precompute-then-serve); on a miss it computes and caches with
a TTL to the next market close.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from .cache import PredictionCache, seconds_to_next_close
from .inference import InsufficientHistory, Predictor, UnknownTicker
from .schemas import HealthResponse, PredictRequest, PredictResponse

# --- Prometheus metrics ----------------------------------------------------
PRED_TOTAL = Counter("predictions_total", "Total prediction requests", ["ticker", "bucket"])
CACHE_HITS = Counter("cache_hits_total", "Prediction cache hits")
CACHE_MISSES = Counter("cache_misses_total", "Prediction cache misses")
ERRORS = Counter("prediction_errors_total", "Prediction errors", ["kind"])
LATENCY = Histogram("predict_latency_seconds", "End-to-end /predict latency",
                    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0))
UP = Gauge("model_up", "1 if the model is loaded")

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["predictor"] = Predictor()
    STATE["cache"] = PredictionCache()
    UP.set(1)
    yield
    UP.set(0)
    STATE.clear()


app = FastAPI(title="NSE Volatility-Bucket Forecaster", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    pred: Predictor | None = STATE.get("predictor")
    cache: PredictionCache | None = STATE.get("cache")
    return HealthResponse(
        status="ok" if pred is not None else "loading",
        model_loaded=pred is not None,
        model_quant=pred.model_quant if pred else "none",
        train_cutoff=pred.train_cutoff if pred else "",
        n_tickers=len(pred.tickers) if pred else 0,
        redis=cache.status if cache else "disabled",
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    pred: Predictor = STATE["predictor"]
    cache: PredictionCache = STATE["cache"]
    start = time.perf_counter()
    try:
        # resolve as-of date first so the cache key matches the served day
        result = pred.predict(req.ticker, req.date)
        key_date = result["as_of_date"]

        hit = cache.get(req.ticker, key_date)
        if hit is not None:
            CACHE_HITS.inc()
            PRED_TOTAL.labels(req.ticker, hit["bucket"]).inc()
            return PredictResponse(**hit, cached=True)

        CACHE_MISSES.inc()
        cache.set(req.ticker, key_date, result, ttl=seconds_to_next_close())
        PRED_TOTAL.labels(req.ticker, result["bucket"]).inc()
        return PredictResponse(**result, cached=False)
    except UnknownTicker as e:
        ERRORS.labels("unknown_ticker").inc()
        raise HTTPException(status_code=404, detail=f"unknown ticker: {e}") from e
    except InsufficientHistory as e:
        ERRORS.labels("insufficient_history").inc()
        raise HTTPException(status_code=422, detail=str(e)) from e
    finally:
        LATENCY.observe(time.perf_counter() - start)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
