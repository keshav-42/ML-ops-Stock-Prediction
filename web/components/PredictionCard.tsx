"use client";

import { BUCKET_COLOR, BUCKET_LABEL, type Bucket, type Prediction } from "@/lib/api";

const ORDER: Bucket[] = ["low", "med", "high"];

export default function PredictionCard({
  prediction,
  loading,
  error,
}: {
  prediction: Prediction | null;
  loading: boolean;
  error: string | null;
}) {
  if (loading) {
    return (
      <div className="flex h-full min-h-[180px] items-center justify-center rounded-xl border border-edge bg-panel">
        <span className="text-sm text-muted">Running model…</span>
      </div>
    );
  }
  if (error) {
    return (
      <div className="flex h-full min-h-[180px] items-center justify-center rounded-xl border border-edge bg-panel px-6">
        <span className="text-sm text-rose-400">{error}</span>
      </div>
    );
  }
  if (!prediction) return null;

  const color = BUCKET_COLOR[prediction.bucket];

  return (
    <div className="rounded-xl border border-edge bg-panel p-6">
      <div className="flex items-baseline justify-between">
        <p className="text-xs uppercase tracking-widest text-muted">
          Next-day volatility · {prediction.predicted_for}
        </p>
        {prediction.cached && (
          <span className="rounded-full border border-edge px-2 py-0.5 text-[10px] uppercase tracking-wider text-muted">
            cached
          </span>
        )}
      </div>

      <p className="mt-3 text-5xl font-semibold tracking-tight" style={{ color }}>
        {BUCKET_LABEL[prediction.bucket]}
      </p>

      <div className="mt-6 space-y-2.5">
        {ORDER.map((b) => {
          const p = prediction.probs[b];
          const isTop = b === prediction.bucket;
          return (
            <div key={b} className="flex items-center gap-3">
              <span className={`w-14 text-xs ${isTop ? "text-foreground" : "text-muted"}`}>
                {BUCKET_LABEL[b]}
              </span>
              <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-zinc-800">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: `${(p * 100).toFixed(1)}%`,
                    background: BUCKET_COLOR[b],
                    opacity: isTop ? 1 : 0.35,
                  }}
                />
              </div>
              <span
                className={`w-12 text-right font-mono text-xs tabular-nums ${
                  isTop ? "text-foreground" : "text-muted"
                }`}
              >
                {(p * 100).toFixed(1)}%
              </span>
            </div>
          );
        })}
      </div>

      <p className="mt-5 text-xs text-muted">
        As of {prediction.as_of_date} · TCN {prediction.model_quant}
      </p>
    </div>
  );
}
