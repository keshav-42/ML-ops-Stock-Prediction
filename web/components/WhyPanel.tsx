"use client";

import { BUCKET_COLOR, BUCKET_LABEL, type Explanation } from "@/lib/api";
import { featureLabel } from "@/lib/features";

const TOP_K = 8;

/** Occlusion attribution: which features pushed the model toward (colored)
 *  or away from (gray) its predicted bucket. */
export default function WhyPanel({ explanation }: { explanation: Explanation | null }) {
  if (!explanation) return null;

  const top = explanation.attributions.slice(0, TOP_K);
  const maxAbs = Math.max(...top.map((a) => Math.abs(a.contribution)), 1e-9);
  const color = BUCKET_COLOR[explanation.bucket];

  return (
    <div className="rounded-xl border border-edge bg-panel p-6">
      <p className="text-xs uppercase tracking-widest text-muted">
        Why {BUCKET_LABEL[explanation.bucket].toLowerCase()}?
      </p>

      <div className="mt-4 space-y-2">
        {top.map((a) => {
          const toward = a.contribution >= 0;
          const width = (Math.abs(a.contribution) / maxAbs) * 100;
          return (
            <div
              key={a.feature}
              className="flex items-center gap-3"
              title={`${a.feature} = ${a.value.toFixed(4)} · ${
                toward ? "+" : "−"
              }${(Math.abs(a.contribution) * 100).toFixed(1)} pp on P(${explanation.bucket})`}
            >
              <span className="w-40 truncate text-xs text-zinc-400">
                {featureLabel(a.feature)}
              </span>
              <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-zinc-800">
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${width.toFixed(1)}%`,
                    background: toward ? color : "#52525b",
                  }}
                />
              </div>
              <span
                className={`w-12 text-right font-mono text-[11px] tabular-nums ${
                  toward ? "text-foreground" : "text-muted"
                }`}
              >
                {toward ? "+" : "−"}
                {(Math.abs(a.contribution) * 100).toFixed(1)}
              </span>
            </div>
          );
        })}
      </div>

      <p className="mt-4 text-[11px] leading-relaxed text-muted">
        Occlusion attribution: each bar is the change (in percentage points) in the
        predicted bucket&apos;s probability when that feature is replaced by its
        historical average. Colored = pushed toward the call, gray = pushed against.
      </p>
    </div>
  );
}
