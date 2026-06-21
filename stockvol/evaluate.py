"""Shared evaluation: walk-forward folds + classification metrics.

Both the LightGBM baseline and the PyTorch TCN report through here on the SAME
folds, so the comparison is honest (HARD RULE 6). We report macro-F1 and per-class
recall — never bare accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from .labels import INT_TO_LABEL
from .splits import Fold, make_walk_forward_folds

CLASSES = [0, 1, 2]


@dataclass
class FoldResult:
    fold: int
    macro_f1: float
    accuracy: float
    recall_per_class: dict[str, float]
    n_train: int
    n_val: int
    confusion: list[list[int]]


@dataclass
class EvalReport:
    model: str
    folds: list[FoldResult] = field(default_factory=list)

    @property
    def mean_macro_f1(self) -> float:
        return float(np.mean([f.macro_f1 for f in self.folds]))

    @property
    def std_macro_f1(self) -> float:
        return float(np.std([f.macro_f1 for f in self.folds]))

    def mean_recall(self) -> dict[str, float]:
        out = {}
        for c in CLASSES:
            name = INT_TO_LABEL[c]
            out[name] = float(np.mean([f.recall_per_class[name] for f in self.folds]))
        return out

    def summary(self) -> str:
        lines = [
            f"=== {self.model} : walk-forward over {len(self.folds)} folds ===",
            f"macro-F1: {self.mean_macro_f1:.4f} ± {self.std_macro_f1:.4f}",
            f"mean per-class recall: "
            + ", ".join(f"{k}={v:.3f}" for k, v in self.mean_recall().items()),
            "",
            f"{'fold':>4} {'n_train':>8} {'n_val':>7} {'macroF1':>8} {'acc':>6}  recall(low/med/high)",
        ]
        for f in self.folds:
            r = f.recall_per_class
            lines.append(
                f"{f.fold:>4} {f.n_train:>8} {f.n_val:>7} {f.macro_f1:>8.4f} "
                f"{f.accuracy:>6.3f}  {r['low']:.3f}/{r['med']:.3f}/{r['high']:.3f}"
            )
        return "\n".join(lines)


def score_fold(
    fold: int, y_true: np.ndarray, y_pred: np.ndarray, n_train: int
) -> FoldResult:
    macro = f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)
    acc = float((y_true == y_pred).mean())
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    return FoldResult(
        fold=fold,
        macro_f1=float(macro),
        accuracy=acc,
        recall_per_class={INT_TO_LABEL[c]: float(rec[i]) for i, c in enumerate(CLASSES)},
        n_train=n_train,
        n_val=len(y_true),
        confusion=cm.tolist(),
    )


def dataset_folds(
    df: pd.DataFrame, n_folds: int = 5, purge_gap: int = 1, min_train: int = 1000
) -> list[Fold]:
    """Walk-forward folds on the dataset's global unique trading dates."""
    return make_walk_forward_folds(
        df["date"], n_folds=n_folds, purge_gap=purge_gap, min_train=min_train
    )
