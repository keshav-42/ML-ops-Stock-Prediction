# data_spec.md — Feature & Label Specification

**Project:** Next-day volatility-bucket forecasting for NSE stocks (3-class: low / med / high).
**Frequency:** daily bars. **Granularity:** one row per `(ticker, date)`.
**Purpose of this file:** human-auditable ground truth for every feature and label. If code
and this spec disagree, this spec wins until explicitly revised. Every definition below
states *exactly which days it reads* so leakage can be verified by inspection.

Notation: for day `t`, `O_t H_t L_t C_t V_t` are open/high/low/close/volume. `ln` is natural
log. All prices are split/dividend-adjusted before anything else runs. Everything is computed
**per ticker, within that ticker's own series** — rolling windows and thresholds never span
two tickers.

---

## 1. The label (this is where projects silently cheat — read carefully)

### 1.1 Per-day volatility estimator (Garman–Klass)

We estimate each day's volatility from its own OHLC using the single-day Garman–Klass
variance:

```
GK_var_t = 0.5 * (ln(H_t / L_t))^2  -  (2*ln(2) - 1) * (ln(C_t / O_t))^2
```

where `2*ln(2) - 1 ≈ 0.386294`.

```
GK_vol_t = sqrt( max(GK_var_t, 0) )
```

**Edge case (must handle):** single-day GK variance can go slightly negative when the
open→close move is large relative to the high→low range. Floor at 0 *before* the sqrt (the
`max(..., 0)` above). Do not silently produce NaN.

**Reads:** `O_t, H_t, L_t, C_t` — all from day `t` only. Using `GK_vol` *for day t* as a
feature is causal and allowed. The **label** uses `GK_vol` for day **t+1** (next section).
Do not confuse the two.

### 1.2 Trailing-tercile bucketing (the causal part)

The label for row `t` is the bucket that **next day's** volatility falls into, measured
against thresholds that were knowable **at the end of day t**:

1. Compute trailing thresholds from a window ending at `t` (window `W = 252` trading days):
   ```
   window  = { GK_vol_{t-W+1}, ..., GK_vol_t }        # ends at t, does NOT include t+1
   q33_t   = 33rd percentile of window
   q66_t   = 66th percentile of window
   ```
2. Assign the label using next-day realized vol:
   ```
   label_t = low   if GK_vol_{t+1} <= q33_t
           = med   if q33_t < GK_vol_{t+1} <= q66_t
           = high  if GK_vol_{t+1} >  q66_t
   ```

**Why this is leakage-free:** thresholds use only days `<= t`; the realized value being
bucketed is `t+1` (the genuine future target). The features on row `t` (Section 2) also use
only days `<= t`. So a complete row `t` = *(information available at close of day t)* →
*(what volatility actually did on t+1)*. That is exactly the supervised setup you want.

**Common wrong version to reject:** computing q33/q66 from the *entire* history (or from a
fixed global quantile over all dates). That leaks future distribution into past labels and
inflates every metric. If you ever see global `quantile()` over the full series for
thresholds, it is a bug.

### 1.3 Rows that must be dropped

- First `max(max_feature_lookback, W)` rows per ticker: insufficient history → drop.
- The final row per ticker has no `t+1` → **no label**; keep it only as the live-inference
  input, never in train/val/test.

---

## 2. Features (all causal: row `t` reads only days `<= t`)

`r_t = ln(C_t / C_{t-1})` is the daily log return (reads `C_t, C_{t-1}` → OK at `t`).

| Feature | Formula (for day `t`) | Reads | Leakage note |
|---|---|---|---|
| `ret_1, ret_5, ret_10, ret_21` | `ln(C_t / C_{t-k})` for k in {1,5,10,21} | `C_t, C_{t-k}` | causal |
| `rstd_5, rstd_10, rstd_21` | rolling std of `r` over last k days ending at `t` | `r_{t-k+1..t}` | causal; trailing window only |
| `parkinson_t` | `sqrt( (1/(4*ln2)) * (ln(H_t/L_t))^2 )` | `H_t, L_t` | causal; `1/(4*ln2) ≈ 0.36067` |
| `gk_t` | `GK_vol_t` (Section 1.1) | `O_t,H_t,L_t,C_t` | causal; **this is a feature, not the label** |
| `gk_ma_5, gk_ma_10, gk_ma_21` | rolling mean of `gk` over last k days | `gk_{t-k+1..t}` | causal |
| `atr_14` | Wilder ATR / `C_t`, where `TR_t = max(H_t-L_t, |H_t-C_{t-1}|, |L_t-C_{t-1}|)` | `H_t,L_t,C_t,C_{t-1}` | causal; normalize by close so it's scale-free |
| `rsi_14` | Wilder RSI(14) on `r` | `r` through `t` | causal |
| `macd, macd_signal, macd_hist` | EMA12−EMA26 of close; signal=EMA9(macd); hist=macd−signal | close through `t` | **use causal EMAs only** (`adjust=False`, no centered windows) |
| `vol_z_21` | `(V_t - mean(V_{t-20..t})) / std(V_{t-20..t})` | `V_{t-20..t}` | causal |
| `nifty_ret_1, nifty_rstd_21` | same as `ret_1`/`rstd_21` on the NIFTY index | NIFTY close through `t` | join on the **same date t** only — never NIFTY `t+1` |
| `vix_level, vix_chg` | India VIX close at `t`; `VIX_t - VIX_{t-1}` | VIX `t, t-1` | causal — VIX is published end-of-day; strongest single vol predictor |
| `dow_sin, dow_cos` | cyclical encoding of day-of-week | calendar | deterministic, no leakage |
| `month_sin, month_cos` | cyclical encoding of month | calendar | deterministic |
| `expiry_week_flag` | 1 if `t` is in the monthly F&O expiry week, else 0 | calendar | deterministic (see note) |
| `days_to_expiry` | trading days from `t` to next monthly expiry | calendar | deterministic |

**Expiry note (verify against a real NSE calendar):** monthly F&O expiry is the **last
Thursday** of the month, rolled back to the prior trading day if that Thursday is a holiday.
NSE has changed weekly-expiry conventions over time; this spec uses the *monthly* rule only.
Compute from an exchange trading calendar, not by assuming Mon–Fri.

**Indicator-library caution:** if using `pandas_ta`/`ta`, verify each indicator uses only
trailing data (no `center=True`, no future-filled NaNs). The MACD/RSI/EMA family is causal
when configured correctly; confirm rather than assume.

---

## 3. Alignment & missing data

- Reindex every series to a shared NSE trading calendar; align stock, NIFTY, and VIX by date.
- **Do not forward-fill prices** to invent bars — drop days a ticker didn't trade. (Ffill on
  prices fabricates returns of zero and corrupts vol.) Small VIX gaps may be ffilled at most
  1 day, and that choice must be logged.
- All `ln(·)` inputs must be strictly positive; assert no non-positive prices survive cleaning.

---

## 4. Global leakage invariants (the leakage-test suite must enforce all of these)

1. **No future bleed.** For each feature `f`, computing `f` on `data[:t]` equals computing
   `f` on the full series then slicing to `[:t]`. (Truncation-invariance test.)
2. **No future columns.** No feature references `*_{t+k}` for `k > 0`. The only `t+1` quantity
   in the entire dataset is the realized vol used to form the label.
3. **Causal thresholds.** Label thresholds derive only from days `<= t` (Section 1.2).
4. **Causal scaling.** Any scaler/normalizer is `fit` on the training fold only; `transform`
   is applied to val/test. Assert fitted params are independent of val/test rows.
5. **Clean splits.** Walk-forward / expanding-window only; never shuffle. Between train-end
   and val-start leave a **purge gap ≥ 1 day** (the horizon) so the last train label (`t+1`)
   cannot fall inside the val feature window.
6. **No cross-ticker bleed.** Every rolling window and quantile is computed strictly within a
   single ticker's series.
7. **Warmup dropped.** Rows lacking full lookback or full threshold window are removed
   (Section 1.3), as is each ticker's final unlabeled row.
8. **Sanity correlation.** No single raw feature is near-perfectly correlated/monotone with
   the label (a flag for accidental target leakage).

A passing run of all eight is the gate to model training. Treat a failure as a stop-the-line
defect, not a warning.

---

## 5. Output schema (the model's training table)

One row per `(ticker, date)`:

```
ticker, date,
ret_1, ret_5, ret_10, ret_21,
rstd_5, rstd_10, rstd_21,
parkinson_t, gk_t, gk_ma_5, gk_ma_10, gk_ma_21,
atr_14, rsi_14, macd, macd_signal, macd_hist, vol_z_21,
nifty_ret_1, nifty_rstd_21, vix_level, vix_chg,
dow_sin, dow_cos, month_sin, month_cos, expiry_week_flag, days_to_expiry,
label            # in {low, med, high} = bucket(GK_vol_{t+1}) vs trailing terciles at t
```

For the PyTorch model, the per-day feature vector above is stacked over a trailing window
(default 30–60 days) into a `(window, n_features)` tensor; the label is the bucket for the
window's last day. The same windowing must respect every invariant in Section 4 (in
particular, windows never cross split boundaries or ticker boundaries).

---

## 6. Implementation notes (Phase 1 — code ↔ spec reconciliation)

These record the concrete choices the code makes. Where this section and code disagree,
this spec still wins (Rule from the header).

- **Wilder ATR/RSI smoothing** is implemented as `ewm(alpha=1/n, adjust=False)` — the exact
  Wilder recursion, causal (no SMA seed warmup gap; warmup is absorbed by the 252-row label
  window drop). MACD uses `ewm(span=…, adjust=False)`. See `stockvol/features.py`.
- **Canonical NSE calendar** = the `^NSEI` (NIFTY) traded dates. Per equity we keep its own
  traded days; NIFTY and VIX context are **left-joined on the equal date `t`**.
- **VIX gaps** are forward-filled at most **1 trading day** and the count is logged in the
  build report (`vix_days_ffilled`; was 17 over 2012–2026). Equity rows still missing NIFTY
  context after the join are **dropped** (`dropped_no_ctx`, ~15/ticker), never fabricated.
- **Calendar features are exempt from the §4.1 truncation-invariance test** and tested
  separately (`test_calendar_features_depend_only_on_date`): they are deterministic
  functions of the date and the *published-in-advance* F&O-expiry + holiday schedule, which
  is knowable at day `t`. They read no prices/volume/VIX, so they carry no market lookahead.
  `days_to_expiry`/`expiry_week_flag` need the forward schedule by construction; the only
  fragile case (an incomplete final month) never occurs in the real build because it uses
  the full NIFTY calendar.
- **Warmup drop** = first **251** rows/ticker (the 252-day threshold window is binding and
  dominates every feature lookback) plus each ticker's final unlabeled (live-inference) row.
- **Output**: `data/processed/features.parquet`, one row per `(ticker, date)`, columns =
  Section-5 schema + `gk_next` (kept for the Phase-5 closed-loop ground-truth check).
  Build: `python -m scripts.build_dataset`. Leakage gate: `pytest tests/test_leakage.py`.
