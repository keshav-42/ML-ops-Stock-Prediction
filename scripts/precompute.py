"""Precompute-then-serve: warm the Redis cache for the latest date, all tickers.

Run post-close (Phase 6 nightly CronJob). Predictions are stable within a trading
day, so this makes /predict a cache hit for every ticker until the next close.

    python -m scripts.precompute
"""

from __future__ import annotations

from stockvol.serving.cache import PredictionCache, seconds_to_next_close
from stockvol.serving.inference import InsufficientHistory, Predictor


def main() -> None:
    predictor = Predictor()
    cache = PredictionCache()
    if cache.status != "connected":
        print(f"WARNING: redis {cache.status}; predictions computed but not cached")

    ttl = seconds_to_next_close()
    ok = 0
    for ticker in predictor.tickers:
        try:
            result = predictor.predict(ticker, date=None)
            cache.set(ticker, result["as_of_date"], result, ttl=ttl)
            ok += 1
            print(f"  {ticker:14s} {result['as_of_date']} -> {result['bucket']:4s} "
                  f"(for {result['predicted_for']})")
        except InsufficientHistory as e:
            print(f"  {ticker:14s} skipped: {e}")

    print(f"\nwarmed {ok}/{len(predictor.tickers)} tickers; ttl={ttl}s "
          f"(redis={cache.status})")


if __name__ == "__main__":
    main()
