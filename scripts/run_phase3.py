"""Phase 3 — transfer learning + INT8 quantization tradeoff table.

    python -m scripts.run_phase3 --target RELIANCE.NS
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from stockvol.config import PROCESSED_DIR
from stockvol.models.tcn import TrainConfig, build_windowed_fold
from stockvol.quantize import benchmark, format_table
from stockvol.transfer import run_transfer, single_temporal_fold

try:
    import mlflow

    _HAVE_MLFLOW = True
except ImportError:  # pragma: no cover
    _HAVE_MLFLOW = False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="RELIANCE.NS")
    p.add_argument("--max-epochs", type=int, default=20)
    args = p.parse_args()

    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    cfg = TrainConfig(max_epochs=args.max_epochs)

    print(f"Phase 3a: transfer learning (target={args.target})")
    result, ft_model = run_transfer(df, args.target, cfg)
    print(result.summary())

    # --- Phase 3b: quantize the fine-tuned model, benchmark on target val ---
    print("\nPhase 3b: INT8 quantization")
    fold = single_temporal_fold(df)
    target_wf = build_windowed_fold(df[df["ticker"] == args.target].copy(), fold)
    rows = benchmark(ft_model, target_wf.Xtr, target_wf.Xva, target_wf.yva)
    table = format_table(rows)
    print(table)

    # persist
    out = {
        "target": args.target,
        "transfer": {
            "from_scratch_f1": result.from_scratch_f1,
            "finetuned_f1": result.finetuned_f1,
            "pooled_zeroshot_f1": result.pooled_zeroshot_f1,
            "improvement": result.improvement,
        },
        "quantization": [
            {"model": r.name, "size_bytes": r.size_bytes,
             "latency_ms_per_sample": r.latency_ms, "macro_f1": r.macro_f1}
            for r in rows
        ],
    }
    path = PROCESSED_DIR / "phase3_results.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")

    if _HAVE_MLFLOW:
        mlflow.set_experiment("volatility-bucket")
        with mlflow.start_run(run_name=f"phase3-{args.target}"):
            mlflow.log_param("target", args.target)
            mlflow.log_metrics({
                "transfer_from_scratch_f1": result.from_scratch_f1,
                "transfer_finetuned_f1": result.finetuned_f1,
                "transfer_improvement": result.improvement,
            })
            for r in rows:
                mlflow.log_metric(f"size_kb_{r.name}", r.size_bytes / 1024)
                mlflow.log_metric(f"latency_ms_{r.name}", r.latency_ms)
                mlflow.log_metric(f"macrof1_{r.name}", r.macro_f1)
            mlflow.log_artifact(str(path))
        print("logged to MLflow (./mlruns)")


if __name__ == "__main__":
    main()
