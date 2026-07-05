# syntax=docker/dockerfile:1
# Multi-stage build -> slim CPU runtime image (serving + batch + monitoring).

# ---- builder: create a venv with all deps (incl. CPU-only torch) ----
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only torch from the PyTorch index (avoids the multi-GB CUDA wheel).
RUN pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements-runtime.txt .
RUN pip install -r requirements-runtime.txt

# ---- runtime: copy venv + code + baked model/feature artifacts ----
FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SERVE_QUANTIZE=1
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY stockvol/ ./stockvol/
COPY scripts/ ./scripts/
# Baked seed data: the trained artifact + the serving feature store. In k8s these
# seed a shared PVC via an initContainer; in compose they are used directly.
COPY data/artifacts/ ./data/artifacts/
COPY data/processed/features.parquet ./data/processed/features.parquet
# Dashboard endpoints: /history reads raw OHLCV, /accuracy reads the live series.
COPY data/raw/ ./data/raw/
COPY data/monitoring/closed_loop.csv ./data/monitoring/closed_loop.csv

# Run as non-root.
RUN useradd -u 10001 -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Container-level healthcheck hits /health (k8s also has its own probes).
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "stockvol.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
