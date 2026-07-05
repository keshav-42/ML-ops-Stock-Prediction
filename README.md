# Next-Day Volatility-Bucket Forecaster (NSE)

An end-to-end ML/MLOps project that predicts the **next-day volatility bucket**
(`low` / `med` / `high`, 3-class) for NSE stocks from daily OHLCV data.

The target is deliberately **volatility, not price direction**. Volatility clusters
and is genuinely forecastable, whereas next-day direction is close to a coin flip —
which would make any monitoring or drift story dishonest. This choice is what lets the
project show a real, measurable model quality signal in production.

> Built as a portfolio project to demonstrate a correct, leakage-free time-series ML
> pipeline all the way from raw data to a containerized, monitored serving stack.

**Author:** Keshav ([keshavtrivedit24@gmail.com](mailto:keshavtrivedit24@gmail.com))

---

## Why this project is different

Most retail "stock ML" projects silently cheat: they shuffle time-series data, scale on
the full dataset, or label with global quantiles that peek into the future. This repo
treats those failure modes as **first-class correctness gates**:

- **Zero lookahead** — every feature for day `t` uses only data `<= t`; labels use `t+1`.
- **Time-series splits only** — walk-forward / expanding window with a **purge gap**
  between train and validation. No shuffling, ever.
- **Causal scaling** — scalers fit on the train fold only (or rolling), never on full data.
- **Trailing-quantile labels** — terciles computed from a trailing 252-day window per
  ticker, never global or forward-looking.
- **Leakage tests gate the models** — the leakage suite must pass before any model code runs.
- **Honest baseline** — LightGBM first, PyTorch TCN compared on the *same* splits;
  metrics reported as macro-F1 + per-class recall, never bare accuracy.

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
