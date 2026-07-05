# VolCast — dashboard

Dark, minimal Next.js UI for the volatility forecaster. Pick an NSE stock, see
its candlestick history, the model's next-day volatility bucket (with class
probabilities), a 60-session predicted-vs-actual hit/miss ribbon, and the live
closed-loop model-health panel.

## Run

The FastAPI serving stack must be up first (from the repo root):

```bash
python -m uvicorn stockvol.serving.app:app --port 8000
```

Then:

```bash
npm install
npm run dev        # http://localhost:3000
```

`NEXT_PUBLIC_API_URL` (see `.env.local`) points at the API; default
`http://localhost:8000`. The API allows the dashboard origin via the
`UI_ORIGINS` env var (default `http://localhost:3000`).

## Endpoints consumed

| Endpoint    | Used for |
|-------------|----------|
| `POST /predict` | next-day bucket + probabilities |
| `GET /history`  | OHLCV candles (TradingView lightweight-charts) |
| `GET /ribbon`   | per-ticker predicted-vs-actual replay |
| `GET /accuracy` | rolling live accuracy / macro-F1 series |
| `GET /health`, `GET /tickers` | status badge + universe |

## Stack

Next.js 16 (App Router) · React 19 · Tailwind v4 · lightweight-charts 5
