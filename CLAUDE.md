# CLAUDE.md — Next-Day Volatility-Bucket Forecaster (NSE)

ML/MLOps portfolio project. Predicts **next-day volatility bucket** (low/med/high, 3-class)
for NSE stocks from daily OHLCV. Target is **volatility, not price direction** — volatility
clusters and is forecastable; daily direction is a coin flip and ruins the monitoring story.

## Stack (pinned — see requirements.txt)
- Python 3.12
- Data: `yfinance` (NSE `.NS` tickers, `^NSEI`, `^INDIAVIX`), `pandas`, `pyarrow` (parquet)
- Models: `lightgbm` (baseline), `torch` 2.8 CPU (TCN)
- Tracking: `mlflow`
- Serving: `fastapi`, `uvicorn`, `pydantic` v2, `redis`
- Monitoring: `prometheus-client`, Grafana, `evidently` (drift)
- Infra: Docker (multi-stage), docker-compose, Kubernetes (kind/minikube → GKE/EKS)

## Layout
```
stockvol/          # library code (small, typed, modular)
  config.py        # pydantic config + ticker universe
  io_utils.py      # parquet read/write
  ingest.py        # Phase 0: fetch OHLCV -> parquet (idempotent)
scripts/           # CLI entrypoints (run_ingest.py, ...)
tests/             # pytest; leakage suite lives here (Phase 1 gate)
data/raw/          # per-ticker parquet (gitignored)
data/processed/    # feature table, splits (gitignored)
CLAUDE.md architecture.md data_spec.md   # context deliverables
```

## Commands
```bash
# install
python -m pip install -r requirements.txt

# Phase 0: ingest (idempotent; safe to re-run / run nightly)
python -m scripts.run_ingest                 # all tickers, default range
python -m scripts.run_ingest --tickers RELIANCE.NS --start 2015-01-01

# models (Phase 2-3)
python -m scripts.train_baseline             # LightGBM walk-forward -> baseline_metrics.json
python -m scripts.train_tcn                  # PyTorch TCN, same folds, MLflow
python -m scripts.run_phase3                 # transfer learning + INT8 quant table

# serving (Phase 4)
python -m scripts.export_model               # train pooled prod model -> data/artifacts/
uvicorn stockvol.serving.app:app --port 8000 # /predict /health /metrics
python -m scripts.precompute                 # warm Redis cache (precompute-then-serve)

# monitoring (Phase 5)
python -m scripts.run_monitoring             # closed-loop accuracy + drift -> data/monitoring/

# docker / k8s (Phase 6)
docker compose up --build                    # api + redis + prometheus + grafana + pushgateway
kubectl apply -k k8s/                        # full stack (see k8s/README.md for kind/minikube)

# tests / lint
pytest -q                                     # 34 tests
pytest tests/test_leakage.py -q              # Phase-1 gate (must pass before models)
python -m ruff check .                        # lint (add ruff in dev)
python -m mypy stockvol                        # types (public fns typed)
```

## Phase plan (STOP after each phase, run checks, summarize, wait for "continue")
- **0 Ingestion** ✅ being built — yfinance OHLCV for ~25 NIFTY-50 + `^NSEI` + `^INDIAVIX`
  → parquet; idempotent; splits via `auto_adjust`. India VIX is a first-class series.
- **1 Features + labels** — all causal features (data_spec.md §2); next-day Garman-Klass
  label with **trailing-tercile** bucketing (§1); write + pass the **leakage suite** (§4).
- **2 Models** — LightGBM baseline FIRST (record walk-forward macro-F1), then PyTorch TCN;
  compare on the SAME splits; MLflow logging.
- **3 Transfer learning + INT8 PTQ** — pretrain pooled encoder, fine-tune per-ticker head;
  dynamic INT8 quantization; size/latency/accuracy table.
- **4 Serving** — FastAPI `/predict` `/health` `/metrics`; Redis cache keyed `(ticker,date)`
  TTL→next close; precompute-then-serve.
- **5 Monitoring** — Prometheus + Grafana; Evidently drift; closed-loop live-accuracy job.
- **6 Docker + K8s** — multi-stage image, compose, Deployment/Service/HPA/ConfigMap/Secret,
  nightly post-close CronJob.

## HARD RULES (non-negotiable — where ML projects silently cheat)
1. **Zero lookahead.** Every feature for day `t` uses only data `<= t`. Labels use `t+1`.
2. **Time-series splits only.** No shuffling. Walk-forward / expanding window with a **purge
   gap** between train and val.
3. **Causal scaling.** Fit scalers on the TRAIN fold only (or rolling). Never on full data.
4. **Trailing-quantile labels.** Terciles from a trailing window (252d), never global/future.
5. **Leakage tests before model code.** They are the correctness gate to Phase 2.
6. **Honest baseline gate.** LightGBM first; PyTorch compared on the same splits. Report
   macro-F1 + per-class recall, never bare accuracy. Every number reproducible from a seed.
7. **PyTorch TCN** over a 30–60d window (small Transformer is an OK alt). Define tensor
   shapes + Dataset interface first.
8. **State assumptions per phase.** One clarifying question if ambiguous, never guess. Never
   invent a library API. Pin versions.
9. **Typed where it counts** (pydantic config + API schemas; type hints on public fns).
   Small modular files.

## Conventions
- `data_spec.md` is the source of truth for features/labels; if code disagrees, spec wins
  until the spec is revised.
- All rolling windows / quantiles computed **strictly within a single ticker's series**.
- Do not forward-fill prices (fabricates zero returns, corrupts vol). VIX gaps: ffill ≤ 1d, logged.
- Reproducibility: fixed `SEED = 42` everywhere stochastic.
