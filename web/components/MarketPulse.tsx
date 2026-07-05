"use client";

import { BUCKET_COLOR, type Bucket, type Prediction } from "@/lib/api";
import { shortSymbol } from "@/lib/tickers";

export type PulseMap = Record<string, Prediction | null>;

function insight(counts: Record<Bucket, number>, total: number): string {
  if (counts.high === 0 && counts.med <= total / 4)
    return `Calm session expected — ${counts.low} of ${total} stocks forecast low volatility.`;
  if (counts.high >= total / 3)
    return `Turbulent session ahead — ${counts.high} of ${total} stocks flagged high volatility.`;
  return `Mixed picture for tomorrow — ${counts.low} low · ${counts.med} medium · ${counts.high} high.`;
}

/** Tomorrow's predicted bucket for the whole universe, as clickable tiles. */
export default function MarketPulse({
  pulse,
  tickers,
  selected,
  onSelect,
}: {
  pulse: PulseMap | null;
  tickers: string[];
  selected: string | null;
  onSelect: (t: string) => void;
}) {
  const loading = pulse === null;
  const preds = Object.values(pulse ?? {}).filter((p): p is Prediction => p !== null);
  const counts: Record<Bucket, number> = { low: 0, med: 0, high: 0 };
  for (const p of preds) counts[p.bucket] += 1;

  return (
    <div className="rounded-xl border border-edge bg-panel p-6">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="text-xs uppercase tracking-widest text-muted">
          Market pulse · tomorrow&apos;s forecast, all {tickers.length} stocks
        </p>
        {!loading && preds.length > 0 && (
          <p className="flex items-center gap-3 font-mono text-xs tabular-nums text-muted">
            {(["low", "med", "high"] as const).map((b) => (
              <span key={b} className="flex items-center gap-1">
                <span
                  className="h-2 w-2 rounded-full"
                  style={{ background: BUCKET_COLOR[b] }}
                />
                {counts[b]}
              </span>
            ))}
          </p>
        )}
      </div>

      {!loading && preds.length > 0 && (
        <p className="mt-2 text-sm text-zinc-300">{insight(counts, preds.length)}</p>
      )}

      <div className="mt-4 grid grid-cols-[repeat(auto-fill,minmax(102px,1fr))] gap-1.5">
        {tickers.map((t) => {
          const p = pulse?.[t] ?? null;
          const active = t === selected;
          if (loading || !p) {
            return (
              <div
                key={t}
                className="h-9 animate-pulse rounded-lg border border-edge bg-zinc-900"
              />
            );
          }
          const color = BUCKET_COLOR[p.bucket];
          return (
            <button
              key={t}
              onClick={() => onSelect(t)}
              title={`${t} → ${p.bucket} (${(p.probs[p.bucket] * 100).toFixed(0)}%)`}
              className={`flex h-9 items-center justify-between rounded-lg border px-2.5 text-xs transition-all hover:-translate-y-0.5 ${
                active ? "border-zinc-500" : "border-edge hover:border-zinc-600"
              }`}
              style={{ background: `${color}14` }}
            >
              <span className={active ? "text-foreground" : "text-zinc-400"}>
                {shortSymbol(t)}
              </span>
              <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: color }} />
            </button>
          );
        })}
      </div>
    </div>
  );
}
