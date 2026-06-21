# architecture.md — Component & Data-Flow Map

Living document. Updated as each phase lands. ✅ = built, 🔜 = planned.

## 1. High-level data flow

```
                          ┌─────────────────────────────────────────────────────┐
                          │  yfinance  (NSE .NS tickers, ^NSEI, ^INDIAVIX)        │
                          └───────────────────────────┬─────────────────────────┘
                                                      │ daily OHLCV (auto-adjusted)
                                                      ▼
   ✅ Phase 0   ingest.py ──► data/raw/<TICKER>.parquet   (idempotent merge, dedup by date)
                                                      │
                                                      ▼
   ✅ Phase 1   features.py + labels.py ─────────────────────────────────────────
                  • align.py: stock/NIFTY/VIX on shared NSE calendar (no ffill of prices;
                    VIX ffill ≤1d logged)                            [data_spec §3]
                  • features.py: causal features (returns, Parkinson/GK vol, ATR/RSI/MACD,
                    vol-z, NIFTY regime, India VIX) + calendar_utils.py (calendar+expiry)
                  • labels.py: next-day GK vol → trailing-tercile bucket [data_spec §1]
                  • dataset.py: build (ticker,date) table → data/processed/features.parquet
                    (82,550 rows × 33 cols, 25 equities, 2013-01 .. 2026-06)
                  • leakage suite (tests/test_leakage.py, 14 tests) ◄ GATE to Phase 2 [§4]
                                                      │
                                                      ▼
   ✅ Phase 1b  splits.py — expanding-window walk-forward + purge gap (no shuffle)
                  scaling.py — CausalStandardScaler (fit on train fold only)
                                                      │
                        ┌─────────────────────────────┴──────────────────────────┐
                        ▼                                                          ▼
   ✅ Phase 2  models/lgbm_baseline.py                          models/tcn.py (PyTorch)
                 macro-F1 0.4159±0.024 (HONEST BASELINE)         (B,28,40) dilated causal TCN
                        │   evaluate.py: shared folds + macro-F1/recall   macro-F1 0.4410±0.027
                        └─────────────────────────────┬──────────────────────────┘
                                                      ▼   TCN beats baseline +0.025 (same folds)
                                          MLflow (params / metrics / artifacts) -> ./mlruns
                                                      │
                                                      ▼
   ✅ Phase 3  transfer.py: pretrain pooled encoder (24 tickers) → freeze → fine-tune
                head on held-out RELIANCE.  scratch 0.4057 < zero-shot 0.4224 <
                finetune 0.4376  (finetune beats from-scratch +0.032 macro-F1)
                quantize.py (torch.ao): FP32 138KB / dyn-INT8 138KB (no-op on conv) /
                static-INT8 56KB (2.45x smaller, 1.14x faster, -0.023 F1)
                                                      │
                                                      ▼  quantized model artifact
   ✅ Phase 4  serving/app.py (FastAPI, lifespan-loaded once)
                 /predict (ticker → bucket + probs)   /health   /metrics (Prometheus)
                 inference.py: static-INT8 model + scaler + feature-store windows
                 artifact.py: FP32 weights + scaler + calib (scripts.export_model)
                 cache.py: Redis key=pred:(ticker,date) TTL→next close (10:00 UTC),
                           graceful degrade if Redis down
                 scripts.precompute warms cache for latest date (precompute-then-serve)
                                                      │
                  ┌───────────────────────────────────┼───────────────────────────────┐
                  ▼                                    ▼                                ▼
   ✅ Phase 5  monitoring/prometheus.yml +    drift.py: manual PSI (gauges) +   closed_loop.py: replay
                alerts.yml + grafana/          Evidently HTML report; 21/28      post-cutoff OOS; 8300 preds,
                dashboards (QPS, p95, errors,  feats drifted, pred-dist PSI      acc 0.499 / F1 0.458;
                cache, live_acc, decay, PSI)   0.203 (med bucket collapsed)      rolling min 0.345 -> DECAY
                scripts.run_monitoring -> metrics.prom (+ optional pushgateway)  ALERT fired. metrics_export.py
                                                      │
                                                      ▼
   ✅ Phase 6  Dockerfile (multi-stage, CPU torch, non-root) → docker-compose
                (api+redis+prometheus+grafana+pushgateway)
                k8s/: ConfigMap, Secret, PVC(seeded via initContainer), Redis,
                Deployment(2x, /health probes), Service, HPA(CPU 70%, 2-6),
                CronJob(30 10 * * 1-5: ingest→build→precompute→monitoring)
                kind/minikube locally; same manifests → GKE/EKS (RWX PVC + LB)
```

## 2. Phase 0 detail (current)

```
config.py
  ├─ TICKER_UNIVERSE: 25 NIFTY-50 equities (.NS)
  ├─ INDEX_TICKER = "^NSEI"      (NIFTY 50 index — market-regime context)
  ├─ VIX_TICKER   = "^INDIAVIX"  (India VIX — first-class vol feature)
  └─ IngestConfig (pydantic): start, end, raw_dir, auto_adjust, max_retries

io_utils.py
  ├─ write_parquet(df, path)      # atomic-ish: write tmp then replace
  └─ read_parquet(path) -> df

ingest.py
  ├─ fetch_one(ticker, start, end, cfg) -> DataFrame   # yfinance.download, retries
  ├─ normalize_ohlcv(df) -> DataFrame                  # canonical cols, tz-naive date
  ├─ merge_idempotent(existing, new) -> DataFrame      # union dates, dedup keep-last
  └─ ingest_ticker / ingest_universe                   # orchestration + manifest

data/raw/<TICKER>.parquet   columns: date, open, high, low, close, volume,
                                     adj_close (==close when auto_adjust), ticker, asset_class
data/raw/_manifest.json     per-ticker: rows, first_date, last_date, fetched_at_utc
```

**Idempotency contract:** re-running ingest fetches from `last_date - lookback_buffer`
forward, merges into the existing parquet, dedups on `date` keeping the latest fetch.
No duplicate rows; running twice in a row is a no-op on data.

## 3. Leakage-control surface (enforced from Phase 1)
- Per-ticker isolation: every window/quantile within one ticker's series.
- Causal-only transforms; truncation-invariance tested.
- Trailing-tercile labels; purge gap on splits; train-only scaler fit.
- See `data_spec.md` §4 for the full invariant list the test suite enforces.
