"use client";

import { useMemo, useState } from "react";
import { metaOf, shortSymbol } from "@/lib/tickers";

interface Props {
  tickers: string[];
  selected: string | null;
  onSelect: (ticker: string) => void;
}

export default function TickerPicker({ tickers, selected, onSelect }: Props) {
  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return tickers;
    return tickers.filter((t) => {
      const m = metaOf(t);
      return (
        t.toLowerCase().includes(q) ||
        m.name.toLowerCase().includes(q) ||
        m.sector.toLowerCase().includes(q)
      );
    });
  }, [tickers, query]);

  return (
    <div className="flex h-full flex-col">
      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search stocks…"
        spellCheck={false}
        className="mb-3 w-full rounded-lg border border-edge bg-panel px-3 py-2 text-sm text-foreground placeholder-muted outline-none transition-colors focus:border-zinc-600"
      />
      <div className="thin-scroll -mr-1 flex-1 space-y-0.5 overflow-y-auto pr-1">
        {filtered.map((t) => {
          const m = metaOf(t);
          const active = t === selected;
          return (
            <button
              key={t}
              onClick={() => onSelect(t)}
              className={`group flex w-full items-center justify-between rounded-lg px-3 py-2 text-left transition-colors ${
                active ? "bg-zinc-800/70" : "hover:bg-zinc-900"
              }`}
            >
              <span>
                <span className="block text-sm font-medium tracking-tight">
                  {shortSymbol(t)}
                </span>
                <span className="block truncate text-xs text-muted">{m.name}</span>
              </span>
              <span className="ml-2 shrink-0 rounded-full border border-edge px-2 py-0.5 text-[10px] uppercase tracking-wider text-muted">
                {m.sector}
              </span>
            </button>
          );
        })}
        {filtered.length === 0 && (
          <p className="px-3 py-6 text-center text-sm text-muted">No matches</p>
        )}
      </div>
    </div>
  );
}
