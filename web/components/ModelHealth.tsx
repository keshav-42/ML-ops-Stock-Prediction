"use client";

import type { AccuracyPoint, Health } from "@/lib/api";

/** Compact strip surfacing the MLOps story: model status + live closed-loop metrics. */
export default function ModelHealth({
  health,
  accuracy,
}: {
  health: Health | null;
  accuracy: { series: AccuracyPoint[]; latest: AccuracyPoint | null } | null;
}) {
  const ok = health?.model_loaded ?? false;
  const latest = accuracy?.latest ?? null;
  const series = (accuracy?.series ?? [])
    .map((p) => p.rolling_acc)
    .filter((v): v is number => v != null);

  return (
    <div className="rounded-xl border border-edge bg-panel p-6">
      <p className="text-xs uppercase tracking-widest text-muted">Model health · live</p>

      <div className="mt-4 flex items-center gap-2">
        <span
          className={`h-2 w-2 rounded-full ${ok ? "bg-emerald-400" : "bg-rose-400"}`}
        />
        <span className="text-sm">{ok ? "Serving" : "Offline"}</span>
        {health && (
          <span className="ml-auto rounded-full border border-edge px-2 py-0.5 text-[10px] uppercase tracking-wider text-muted">
            {health.model_quant}
          </span>
        )}
      </div>

      {latest && (
        <div className="mt-4 grid grid-cols-2 gap-4">
          <div>
            <p className="font-mono text-2xl tabular-nums">
              {latest.rolling_acc != null ? (latest.rolling_acc * 100).toFixed(1) : "—"}
              <span className="text-sm text-muted">%</span>
            </p>
            <p className="text-xs text-muted">21-day rolling accuracy</p>
          </div>
          <div>
            <p className="font-mono text-2xl tabular-nums">
              {latest.rolling_f1 != null ? latest.rolling_f1.toFixed(3) : "—"}
            </p>
            <p className="text-xs text-muted">rolling macro-F1</p>
          </div>
        </div>
      )}

      {series.length > 1 && <Sparkline values={series} />}

      {health && (
        <p className="mt-4 text-xs text-muted">
          {health.n_tickers} tickers · trained through {health.train_cutoff} · Redis{" "}
          {health.redis}
        </p>
      )}
    </div>
  );
}

function Sparkline({ values }: { values: number[] }) {
  const w = 260;
  const h = 40;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const pts = values
    .map(
      (v, i) =>
        `${((i / (values.length - 1)) * w).toFixed(1)},${(h - ((v - min) / span) * h).toFixed(1)}`,
    )
    .join(" ");

  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      className="mt-4 h-10 w-full"
      preserveAspectRatio="none"
      aria-label="rolling accuracy trend"
    >
      <polyline
        points={pts}
        fill="none"
        stroke="#34d399"
        strokeWidth="1.5"
        strokeLinejoin="round"
        vectorEffect="non-scaling-stroke"
        opacity="0.8"
      />
    </svg>
  );
}
