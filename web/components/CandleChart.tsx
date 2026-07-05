"use client";

import { useEffect, useRef } from "react";
import {
  CandlestickSeries,
  ColorType,
  HistogramSeries,
  createChart,
} from "lightweight-charts";
import type { Candle } from "@/lib/api";

export default function CandleChart({ candles }: { candles: Candle[] }) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el || candles.length === 0) return;

    const chart = createChart(el, {
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#71717a",
        fontFamily: "var(--font-geist-sans), system-ui, sans-serif",
        attributionLogo: false,
      },
      grid: {
        vertLines: { color: "rgba(39,39,42,0.5)" },
        horzLines: { color: "rgba(39,39,42,0.5)" },
      },
      rightPriceScale: { borderColor: "#26262b" },
      timeScale: { borderColor: "#26262b", rightOffset: 3 },
      crosshair: {
        horzLine: { labelBackgroundColor: "#3f3f46" },
        vertLine: { labelBackgroundColor: "#3f3f46" },
      },
    });

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#34d399",
      downColor: "#fb7185",
      wickUpColor: "#34d399",
      wickDownColor: "#fb7185",
      borderVisible: false,
    });
    candleSeries.setData(
      candles.map((c) => ({
        time: c.date,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
      })),
    );

    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceScaleId: "volume",
      priceFormat: { type: "volume" },
      color: "#3f3f46",
    });
    volumeSeries.priceScale().applyOptions({
      scaleMargins: { top: 0.82, bottom: 0 },
    });
    volumeSeries.setData(
      candles.map((c) => ({
        time: c.date,
        value: c.volume,
        color: c.close >= c.open ? "rgba(52,211,153,0.25)" : "rgba(251,113,133,0.25)",
      })),
    );

    chart.timeScale().fitContent();
    return () => chart.remove();
  }, [candles]);

  return <div ref={containerRef} className="h-[360px] w-full" />;
}
