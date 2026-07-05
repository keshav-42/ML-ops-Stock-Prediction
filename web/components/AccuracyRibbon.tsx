"use client";

import { BUCKET_COLOR, BUCKET_LABEL, type RibbonResponse } from "@/lib/api";

/** Last-N-days hit/miss strip: solid square = model got that day right,
 *  hollow = miss. Color = the ACTUAL bucket, so misses show what really happened. */
export default function AccuracyRibbon({ ribbon }: { ribbon: RibbonResponse | null }) {
  if (!ribbon || ribbon.entries.length === 0) return null;

  return (
    <div className="rounded-xl border border-edge bg-panel p-6">
      <div className="flex items-baseline justify-between">
        <p className="text-xs uppercase tracking-widest text-muted">
          Predicted vs actual · last {ribbon.entries.length} sessions
        </p>
        {ribbon.hit_rate != null && (
          <p className="font-mono text-sm tabular-nums">
            {(ribbon.hit_rate * 100).toFixed(0)}%{" "}
            <span className="text-xs text-muted">hit rate</span>
          </p>
        )}
      </div>

      <div className="mt-4 flex flex-wrap gap-1">
        {ribbon.entries.map((e) => {
          const hit = e.predicted === e.actual;
          const color = BUCKET_COLOR[e.actual];
          return (
            <span
              key={e.date}
              title={`${e.date} · predicted ${BUCKET_LABEL[e.predicted]}, actual ${BUCKET_LABEL[e.actual]} — ${hit ? "hit" : "miss"}`}
              className="h-4 w-4 rounded-[4px] transition-transform hover:scale-125"
              style={
                hit
                  ? { background: color }
                  : { border: `1.5px solid ${color}`, opacity: 0.55 }
              }
            />
          );
        })}
      </div>

      <div className="mt-4 flex items-center gap-5 text-xs text-muted">
        <span className="flex items-center gap-1.5">
          <span className="h-3 w-3 rounded-[3px] bg-zinc-400" /> hit
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-3 w-3 rounded-[3px] border-[1.5px] border-zinc-400 opacity-60" />{" "}
          miss (color = actual)
        </span>
        {(["low", "med", "high"] as const).map((b) => (
          <span key={b} className="flex items-center gap-1.5">
            <span className="h-3 w-3 rounded-[3px]" style={{ background: BUCKET_COLOR[b] }} />
            {BUCKET_LABEL[b]}
          </span>
        ))}
      </div>
    </div>
  );
}
