"""Production model artifact: FP32 weights + scaler + metadata + calibration set.

We persist FP32 weights and static-quantize at load time (the quantized module is
rebuilt deterministically from a saved calibration sample). This avoids fragile
serialization of quantized modules while still serving INT8 (Phase 3 result).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..config import REPO_ROOT

ARTIFACT_DIR = REPO_ROOT / "data" / "artifacts"


@dataclass
class ProductionArtifact:
    feature_columns: list[str]
    classes: list[str]
    window: int
    channels: list[int]
    train_cutoff: str
    created_at: str
    scaler_mean: list[float]
    scaler_std: list[float]

    def to_meta(self) -> dict:
        return {
            "feature_columns": self.feature_columns,
            "classes": self.classes,
            "window": self.window,
            "channels": self.channels,
            "train_cutoff": self.train_cutoff,
            "created_at": self.created_at,
        }


def save_artifact(
    artifact: ProductionArtifact,
    state_dict: dict,
    calib: np.ndarray,
    out_dir: Path = ARTIFACT_DIR,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, out_dir / "model_fp32.pt")
    np.save(out_dir / "calib.npy", calib.astype(np.float32))
    (out_dir / "meta.json").write_text(json.dumps(artifact.to_meta(), indent=2), "utf-8")
    (out_dir / "scaler.json").write_text(
        json.dumps(
            {"columns": artifact.feature_columns,
             "mean": artifact.scaler_mean, "std": artifact.scaler_std},
            indent=2,
        ),
        "utf-8",
    )
    return out_dir


def load_meta(out_dir: Path = ARTIFACT_DIR) -> dict:
    return json.loads((out_dir / "meta.json").read_text("utf-8"))


def load_scaler(out_dir: Path = ARTIFACT_DIR) -> dict:
    return json.loads((out_dir / "scaler.json").read_text("utf-8"))
