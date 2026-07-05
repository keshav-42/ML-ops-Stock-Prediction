"use client";

export interface IndexStat {
  label: string;
  value: string;
  changePct: number | null;
}

/** Plain-English intro + live market context, so a first-time visitor
 *  immediately knows what the site does. */
export default function Hero({
  stats,
  liveAccuracy,
}: {
  stats: IndexStat[];
  liveAccuracy: number | null;
}) {
  return (
    <section className="grid gap-6 border-b border-edge py-8 lg:grid-cols-[1fr_auto]">
      <div className="max-w-2xl">
        <h2 className="text-3xl font-semibold leading-tight tracking-tight">
          Will your stock be calm or wild{" "}
          <span className="text-emerald-400">tomorrow</span>?
        </h2>
        <p className="mt-3 text-sm leading-relaxed text-zinc-400">
          Every day a neural network reads 28 market signals — realized volatility,
          India VIX, volume, momentum — and forecasts each NSE stock&apos;s next-day
          volatility: <span className="text-emerald-400">low</span>,{" "}
          <span className="text-amber-400">medium</span> or{" "}
          <span className="text-rose-400">high</span>. Then it grades itself in public,
          every single day.
        </p>
        <div className="mt-4 flex flex-wrap items-center gap-2 text-xs text-muted">
          <Step n={1} text="Pick a stock" />
          <Arrow />
          <Step n={2} text="See tomorrow's forecast & why" />
          <Arrow />
          <Step n={3} text="Check its live track record" />
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-3 lg:flex-col lg:items-stretch lg:justify-center">
        {stats.map((s) => (
          <div
            key={s.label}
            className="min-w-36 rounded-xl border border-edge bg-panel px-4 py-3"
          >
            <p className="text-[10px] uppercase tracking-widest text-muted">{s.label}</p>
            <p className="mt-1 font-mono text-lg tabular-nums">
              {s.value}
              {s.changePct != null && (
                <span
                  className={`ml-2 text-xs ${
                    s.changePct >= 0 ? "text-emerald-400" : "text-rose-400"
                  }`}
                >
                  {s.changePct >= 0 ? "+" : ""}
                  {s.changePct.toFixed(2)}%
                </span>
              )}
            </p>
          </div>
        ))}
        {liveAccuracy != null && (
          <div className="min-w-36 rounded-xl border border-edge bg-panel px-4 py-3">
            <p className="text-[10px] uppercase tracking-widest text-muted">
              Live accuracy · 21d
            </p>
            <p className="mt-1 font-mono text-lg tabular-nums">
              {(liveAccuracy * 100).toFixed(1)}%
              <span className="ml-2 text-xs text-muted">vs 33% chance</span>
            </p>
          </div>
        )}
      </div>
    </section>
  );
}

function Step({ n, text }: { n: number; text: string }) {
  return (
    <span className="flex items-center gap-1.5 rounded-full border border-edge px-3 py-1">
      <span className="font-mono text-emerald-400">{n}</span> {text}
    </span>
  );
}

function Arrow() {
  return <span className="text-zinc-600">→</span>;
}
