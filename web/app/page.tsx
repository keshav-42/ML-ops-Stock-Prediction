"use client";

import { useCallback, useEffect, useState } from "react";
import AccuracyRibbon from "@/components/AccuracyRibbon";
import CandleChart from "@/components/CandleChart";
import ModelHealth from "@/components/ModelHealth";
import PredictionCard from "@/components/PredictionCard";
import TickerPicker from "@/components/TickerPicker";
import WhyPanel from "@/components/WhyPanel";
import {
  api,
  type AccuracyPoint,
  type Candle,
  type Explanation,
  type Health,
  type Prediction,
  type RibbonResponse,
} from "@/lib/api";
import { metaOf, shortSymbol } from "@/lib/tickers";

export default function Dashboard() {
  const [tickers, setTickers] = useState<string[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [accuracy, setAccuracy] = useState<{
    series: AccuracyPoint[];
    latest: AccuracyPoint | null;
  } | null>(null);
  const [apiDown, setApiDown] = useState(false);

  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [candles, setCandles] = useState<Candle[]>([]);
  const [ribbon, setRibbon] = useState<RibbonResponse | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [predictError, setPredictError] = useState<string | null>(null);

  // Global data: universe, health, live accuracy.
  useEffect(() => {
    (async () => {
      try {
        const [t, h] = await Promise.all([api.tickers(), api.health()]);
        setTickers(t.tickers);
        setHealth(h);
        setSelected((s) => s ?? (t.tickers.includes("RELIANCE.NS") ? "RELIANCE.NS" : t.tickers[0]));
      } catch {
        setApiDown(true);
        return;
      }
      // Live-accuracy series is optional (absent until monitoring has run once).
      try {
        setAccuracy(await api.accuracy());
      } catch {
        /* panel simply shows less */
      }
    })();
  }, []);

  // Per-ticker data whenever the selection changes.
  const loadTicker = useCallback(async (ticker: string) => {
    setPredicting(true);
    setPredictError(null);
    setPrediction(null);
    setExplanation(null);
    try {
      const [p, h, r, ex] = await Promise.all([
        api.predict(ticker),
        api.history(ticker, 180),
        api.ribbon(ticker, 60),
        api.explain(ticker).catch(() => null),
      ]);
      setPrediction(p);
      setCandles(h.candles);
      setRibbon(r);
      setExplanation(ex);
    } catch (e) {
      setPredictError(e instanceof Error ? e.message : "prediction failed");
    } finally {
      setPredicting(false);
    }
  }, []);

  useEffect(() => {
    if (selected) void loadTicker(selected);
  }, [selected, loadTicker]);

  const meta = selected ? metaOf(selected) : null;
  const last = candles.at(-1);
  const prev = candles.at(-2);
  const dayChange = last && prev ? ((last.close - prev.close) / prev.close) * 100 : null;

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-1 flex-col px-6">
      {/* Top bar */}
      <header className="flex items-center justify-between border-b border-edge py-5">
        <div className="flex items-baseline gap-3">
          <h1 className="text-lg font-semibold tracking-tight">VolCast</h1>
          <p className="hidden text-sm text-muted sm:block">
            next-day volatility forecasts · NSE
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs text-muted">
          <span
            className={`h-1.5 w-1.5 rounded-full ${
              apiDown ? "bg-rose-400" : health?.model_loaded ? "bg-emerald-400" : "bg-amber-400"
            }`}
          />
          {apiDown ? "API offline" : health?.model_loaded ? "model live" : "loading"}
        </div>
      </header>

      {apiDown ? (
        <div className="flex flex-1 items-center justify-center py-32">
          <div className="max-w-md text-center">
            <p className="text-lg font-medium">API is not reachable</p>
            <p className="mt-2 text-sm text-muted">
              Start the serving stack first:{" "}
              <code className="rounded bg-zinc-900 px-1.5 py-0.5 font-mono text-xs">
                uvicorn stockvol.serving.app:app --port 8000
              </code>{" "}
              then reload this page.
            </p>
          </div>
        </div>
      ) : (
        <div className="grid flex-1 gap-6 py-6 lg:grid-cols-[240px_1fr]">
          {/* Sidebar */}
          <aside className="lg:sticky lg:top-6 lg:h-[calc(100vh-140px)]">
            <TickerPicker tickers={tickers} selected={selected} onSelect={setSelected} />
          </aside>

          {/* Main */}
          <main className="min-w-0 space-y-6">
            {selected && meta && (
              <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
                <h2 className="text-2xl font-semibold tracking-tight">
                  {shortSymbol(selected)}
                </h2>
                <p className="text-sm text-muted">{meta.name}</p>
                {last && (
                  <p className="ml-auto font-mono text-sm tabular-nums">
                    ₹{last.close.toLocaleString("en-IN")}
                    {dayChange != null && (
                      <span
                        className={`ml-2 ${dayChange >= 0 ? "text-emerald-400" : "text-rose-400"}`}
                      >
                        {dayChange >= 0 ? "+" : ""}
                        {dayChange.toFixed(2)}%
                      </span>
                    )}
                  </p>
                )}
              </div>
            )}

            <div className="grid gap-6 xl:grid-cols-[1fr_340px]">
              <div className="min-w-0 rounded-xl border border-edge bg-panel p-4">
                {candles.length > 0 ? (
                  <CandleChart candles={candles} />
                ) : (
                  <div className="flex h-[360px] items-center justify-center text-sm text-muted">
                    Loading chart…
                  </div>
                )}
              </div>
              <div className="space-y-6">
                <PredictionCard
                  prediction={prediction}
                  loading={predicting}
                  error={predictError}
                />
                {!predicting && <WhyPanel explanation={explanation} />}
              </div>
            </div>

            <div className="grid gap-6 xl:grid-cols-[1fr_340px]">
              <AccuracyRibbon ribbon={ribbon} />
              <ModelHealth health={health} accuracy={accuracy} />
            </div>
          </main>
        </div>
      )}

      <footer className="border-t border-edge py-4 text-center text-xs text-muted">
        Volatility buckets from trailing terciles — not investment advice. TCN · INT8 ·
        FastAPI · Redis · Prometheus
      </footer>
    </div>
  );
}
