"""Pydantic v2 request/response schemas for the API (typed contract)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    ticker: str = Field(..., examples=["RELIANCE.NS"], description="NSE .NS ticker.")
    date: str | None = Field(
        default=None,
        description="As-of trading date YYYY-MM-DD; default = latest available.",
    )


class PredictResponse(BaseModel):
    ticker: str
    as_of_date: str = Field(..., description="Trading day the window ends on (day t).")
    predicted_for: str = Field(..., description="Trading day being predicted (t+1).")
    bucket: str = Field(..., description="Predicted next-day vol bucket: low|med|high.")
    probs: dict[str, float] = Field(..., description="Class probabilities.")
    cached: bool = Field(..., description="True if served from the Redis cache.")
    model_quant: str = Field(..., description="Quantization of the served model.")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_quant: str
    train_cutoff: str
    n_tickers: int
    redis: str  # "connected" | "unavailable" | "disabled"
