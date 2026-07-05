/** Typed client for the FastAPI serving stack (stockvol.serving). */

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Bucket = "low" | "med" | "high";

export interface Prediction {
  ticker: string;
  as_of_date: string;
  predicted_for: string;
  bucket: Bucket;
  probs: Record<Bucket, number>;
  cached: boolean;
  model_quant: string;
}

export interface Candle {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface RibbonEntry {
  date: string;
  predicted: Bucket;
  actual: Bucket;
}

export interface RibbonResponse {
  ticker: string;
  entries: RibbonEntry[];
  hit_rate: number | null;
}

export interface AccuracyPoint {
  date: string;
  daily_acc: number | null;
  rolling_acc: number | null;
  rolling_f1: number | null;
}

export interface Attribution {
  feature: string;
  contribution: number;
  value: number;
}

export interface Explanation {
  ticker: string;
  as_of_date: string;
  bucket: Bucket;
  prob: number;
  attributions: Attribution[];
}

export interface Health {
  status: string;
  model_loaded: boolean;
  model_quant: string;
  train_cutoff: string;
  n_tickers: number;
  redis: string;
}

async function get<T>(path: string, params?: Record<string, string | number>): Promise<T> {
  const url = new URL(path, BASE);
  for (const [k, v] of Object.entries(params ?? {})) url.searchParams.set(k, String(v));
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail ?? res.statusText);
  return res.json();
}

export const api = {
  health: () => get<Health>("/health"),
  tickers: () => get<{ tickers: string[]; window: number; train_cutoff: string }>("/tickers"),
  history: (ticker: string, days = 180) =>
    get<{ ticker: string; candles: Candle[] }>("/history", { ticker, days }),
  ribbon: (ticker: string, days = 60) => get<RibbonResponse>("/ribbon", { ticker, days }),
  explain: (ticker: string) => get<Explanation>("/explain", { ticker }),
  accuracy: (days = 120) => get<{ series: AccuracyPoint[]; latest: AccuracyPoint | null }>("/accuracy", { days }),
  predict: async (ticker: string): Promise<Prediction> => {
    const res = await fetch(new URL("/predict", BASE), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker }),
    });
    if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail ?? res.statusText);
    return res.json();
  },
};

export const BUCKET_COLOR: Record<Bucket, string> = {
  low: "#34d399", // emerald-400
  med: "#fbbf24", // amber-400
  high: "#fb7185", // rose-400
};

export const BUCKET_LABEL: Record<Bucket, string> = {
  low: "Low",
  med: "Medium",
  high: "High",
};
