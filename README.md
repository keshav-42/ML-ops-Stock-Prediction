# VolCast — Next-Day Volatility Forecaster for NSE Stocks

[![CI](https://github.com/keshav-42/ML-ops-Stock-Prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/keshav-42/ML-ops-Stock-Prediction/actions/workflows/ci.yml)
**[Live demo →](https://ml-ops-stock-prediction.vercel.app)**

An end-to-end ML + MLOps system that forecasts the **next-day volatility bucket**
(`low` / `med` / `high`) for 25 NIFTY-50 stocks — from raw OHLCV ingestion through
a leakage-tested training pipeline, a quantized PyTorch model served behind
FastAPI, live drift/accuracy monitoring, and a Next.js dashboard that explains
every prediction.

**Author:** Keshav ([keshavtrivedit24@gmail.com](mailto:keshavtrivedit24@gmail.com))

---

## Why volatility, not price?

Most retail "stock ML" projects predict price direction — and silently cheat to
make the numbers look good: shuffled time-series splits, scalers fit on the full
dataset, labels that peek into the future. Next-day *direction* is close to a coin
flip, so any impressive accuracy is almost always leakage.

**Volatility is different.** It clusters, it persists, and it is genuinely
forecastable — which means this project can show a *real, measurable* model
quality signal in production and grade itself honestly, every day. The live
rolling accuracy (~37% vs a 33% random baseline for 3 classes) is modest and
**real** — and the entire system is built to keep it that way.

---

## Features

**Honest ML pipeline (the core)**
- **Zero lookahead** — every feature for day `t` uses only data `≤ t`; labels use `t+1`
- **Walk-forward splits with a purge gap** — no shuffling, ever; scalers fit on train folds only
- **Trailing-tercile labels** — computed per ticker from a trailing 252-day window, never global
- **Leakage test suite gates the models** — it must pass before any model code runs (enforced in CI)
- **Honest baseline** — LightGBM first (macro-F1 0.416), PyTorch TCN compared on the *same* splits (0.441); metrics are macro-F1 + per-class recall, never bare accuracy

**Modeling**
- Temporal Convolutional Network over 60-day windows of 28 causal features (incl. India VIX, Parkinson & Garman–Klass range volatility)
- **Transfer learning**: pooled encoder trained across all tickers + fine-tuned heads (+3.2 F1 pts over from-scratch)
- **Static INT8 quantization**: 2.5× smaller, faster on CPU, −0.03 F1 pts (~free)

**Production serving**
- FastAPI `/predict` with Redis **precompute-then-serve** caching (TTL to next market close), Prometheus metrics, p99 latency histograms
- Explainability endpoint: **occlusion attribution** that works directly on the INT8 model

**Monitoring that closes the loop**
- Daily **closed-loop live accuracy**: yesterday's predictions graded against what actually happened, with rolling macro-F1 and a decay alert threshold
- **Evidently drift reports** + prediction-distribution PSI, exported to Prometheus/Grafana

**Dashboard ([live](https://ml-ops-stock-prediction.vercel.app))**
- Next.js 16 dark-minimal UI: market pulse across all 25 stocks, candlestick charts, per-prediction "why" panel, 60-session hit/miss ribbon, live model-health panel

**Infrastructure**
- Multi-stage Docker (CPU-only torch), docker-compose stack (API + Redis + Prometheus + Grafana), Kubernetes manifests (HPA, CronJob, probes), GitHub Actions CI

Feature/label definitions live in [data_spec.md](data_spec.md); system design in
[architecture.md](architecture.md).

---

## Architecture at a glance

```
yfinance OHLCV ──► parquet ──► causal features + trailing-tercile labels
   (NSE .NS,          (raw)          │
    ^NSEI, ^INDIAVIX)                ▼
                          walk-forward splits (purge gap)
                                     │
                    ┌────────────────┴─────────────────┐
              LightGBM baseline               PyTorch TCN (30–60d window)
                    └────────────────┬─────────────────┘
                                     ▼
                 transfer learning (pooled encoder + per-ticker head)
                              + INT8 dynamic quantization
                                     │
                                     ▼
              FastAPI /predict /health /metrics  ◄──► Redis cache
                     (precompute-then-serve)      (keyed by ticker,date)
                                     │
                                     ▼
            Prometheus + Grafana + Evidently drift + closed-loop accuracy
                                     │
                                     ▼
              Docker (multi-stage) · docker-compose · Kubernetes (HPA, CronJob)
```

---

## Tech stack

| Area        | Tools |
|-------------|-------|
| Language    | Python 3.12 |
| Data        | `yfinance`, `pandas`, `pyarrow` (parquet) |
| Models      | `lightgbm` (baseline), `torch` 2.8 CPU (TCN) |
| Tracking    | `mlflow` |
| Serving     | `fastapi`, `uvicorn`, `pydantic` v2, `redis` |
| Monitoring  | `prometheus-client`, Grafana, `evidently` |
| Dashboard   | Next.js 16, React 19, Tailwind v4, TradingView `lightweight-charts` |
| Infra       | Docker (multi-stage), docker-compose, Kubernetes |

---

## Repository layout

```
stockvol/                 # library code (small, typed, modular)
  config.py               # pydantic config + ticker universe
  ingest.py               # Phase 0: yfinance OHLCV -> parquet (idempotent)
  features.py labels.py   # Phase 1: causal features + trailing-tercile labels
  splits.py scaling.py    # walk-forward splits, causal scalers
  dataset.py align.py     # TCN windowing / tensor dataset, series alignment
  models/                 # lgbm_baseline.py, tcn.py
  transfer.py quantize.py # Phase 3: transfer learning + INT8 PTQ
  serving/                # FastAPI app, Redis cache, inference, schemas
  monitoring/             # closed-loop accuracy, drift, metrics export
scripts/                  # CLI entrypoints (run_ingest, train_*, export_model, ...)
tests/                    # pytest (34 tests); leakage suite is the Phase-1 gate
k8s/                      # Deployment/Service/HPA/ConfigMap/Secret/CronJob + README
web/                      # Next.js dashboard (dark, minimal; talks to the API)
data/                     # raw/ processed/ artifacts/ monitoring/ (gitignored)
architecture.md data_spec.md             # design + data-spec deliverables
```

---

## Quick start

```bash
# install
python -m pip install -r requirements.txt

# Phase 0 — ingest (idempotent; safe to re-run nightly)
python -m scripts.run_ingest
python -m scripts.run_ingest --tickers RELIANCE.NS --start 2015-01-01

# Phase 1 — build feature table + labels
python -m scripts.build_dataset

# Phase 2 — models (baseline first, then TCN, same splits, MLflow logging)
python -m scripts.train_baseline    # LightGBM walk-forward -> baseline_metrics.json
python -m scripts.train_tcn         # PyTorch TCN

# Phase 3 — transfer learning + INT8 quantization table
python -m scripts.run_phase3

# Phase 4 — serving
python -m scripts.export_model                    # train pooled prod model -> data/artifacts/
uvicorn stockvol.serving.app:app --port 8000      # /predict /health /metrics
python -m scripts.precompute                       # warm Redis cache

# Phase 5 — monitoring
python -m scripts.run_monitoring                   # closed-loop accuracy + drift

# Dashboard — Next.js UI (needs the API from Phase 4 running on :8000)
cd web && npm install && npm run dev               # http://localhost:3000

# Phase 6 — docker / k8s
docker compose up --build                          # api + redis + prometheus + grafana + pushgateway
kubectl apply -k k8s/                              # full stack (see k8s/README.md)
```

Public deploy (Render free tier for the API + Vercel for the dashboard): see
[DEPLOY.md](DEPLOY.md).

---

## Tests & quality gates

```bash
pytest -q                        # 34 tests
pytest tests/test_leakage.py -q  # Phase-1 gate — must pass before models
python -m ruff check .           # lint
python -m mypy stockvol          # types (public fns typed)
```

The **leakage suite** ([tests/test_leakage.py](tests/test_leakage.py)) is the correctness
gate that unlocks the modeling phases — it asserts no feature uses future data, splits
never overlap without a purge gap, and label quantiles are strictly trailing.

---

## Build phases

The project was built in disciplined, self-contained phases — each is a single commit that
passes its own checks before the next begins.

| Phase | Commit | What it delivers |
|-------|--------|------------------|
| — | `chore: project scaffolding and pinned dependencies` | Repo skeleton, pinned `requirements.txt` |
| — | `docs: context deliverables` | architecture.md, data_spec.md |
| **0** | `feat(ingest)` | Idempotent yfinance → parquet ingestion; India VIX as a first-class series |
| **1** | `feat(features)` | Causal features, trailing-tercile labels, and the leakage suite |
| **2** | `feat(models)` | LightGBM baseline + PyTorch TCN on identical splits |
| **3** | `feat(transfer+quant)` | Transfer learning (pooled encoder + per-ticker head) + INT8 PTQ |
| **4** | `feat(serving)` | FastAPI `/predict` `/health` `/metrics` + Redis precompute-then-serve |
| **5** | `feat(monitoring)` | Evidently drift + closed-loop live-accuracy job |
| **6** | `feat(deploy)` | Multi-stage Docker image, docker-compose, Kubernetes manifests |

---

## Conventions & reproducibility

- [data_spec.md](data_spec.md) is the source of truth for features/labels — if code and
  spec disagree, the spec wins until revised.
- All rolling windows / quantiles are computed **strictly within a single ticker's series**.
- Prices are never forward-filled (that fabricates zero returns and corrupts volatility);
  VIX gaps are ffilled ≤ 1 day and logged.
- Fixed `SEED = 42` everywhere stochastic; every reported number is reproducible from a seed.

---

## License

No license file is currently included. Add one (e.g. MIT) before public reuse.
