"""CLI: build the Phase-1 feature/label table from raw parquet.

    python -m scripts.build_dataset
    python -m scripts.build_dataset --tickers RELIANCE.NS INFY.NS
"""

from __future__ import annotations

import argparse

from stockvol.dataset import build_dataset, write_dataset


def main() -> None:
    p = argparse.ArgumentParser(description="Build features+labels parquet.")
    p.add_argument("--tickers", nargs="*", default=None)
    args = p.parse_args()

    df, report = build_dataset(tickers=args.tickers)
    path = write_dataset(df)

    print("Built dataset:")
    print(report.summary())
    print(f"\nlabel distribution:\n{df['label'].value_counts(normalize=True).round(3)}")
    print(f"\nwrote {len(df)} rows x {df.shape[1]} cols -> {path}")


if __name__ == "__main__":
    main()
