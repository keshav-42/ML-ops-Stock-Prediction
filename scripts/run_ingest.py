"""CLI entrypoint for Phase-0 ingestion.

Examples:
    python -m scripts.run_ingest
    python -m scripts.run_ingest --tickers RELIANCE.NS ^INDIAVIX --start 2015-01-01
"""

from __future__ import annotations

import argparse
from datetime import date

from stockvol.config import IngestConfig
from stockvol.ingest import ingest_universe


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest NSE daily OHLCV to parquet.")
    p.add_argument("--tickers", nargs="*", default=None,
                   help="Subset of tickers (default: full universe).")
    p.add_argument("--start", type=_parse_date, default=None,
                   help="History start YYYY-MM-DD (default 2012-01-01).")
    p.add_argument("--end", type=_parse_date, default=None,
                   help="History end YYYY-MM-DD exclusive (default: today).")
    args = p.parse_args()

    kwargs: dict = {}
    if args.start is not None:
        kwargs["start"] = args.start
    if args.end is not None:
        kwargs["end"] = args.end
    cfg = IngestConfig(**kwargs)

    print(f"Ingesting -> {cfg.raw_dir}  (start={cfg.start}, end={cfg.end or 'today'})")
    manifest = ingest_universe(cfg, tickers=args.tickers)

    ok = sum(1 for v in manifest.values() if "error" not in v)
    print(f"\nDone: {ok}/{len(manifest)} tickers ingested. Manifest: {cfg.raw_dir / '_manifest.json'}")


if __name__ == "__main__":
    main()
