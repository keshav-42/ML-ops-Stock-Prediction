"""Phase 3a — transfer learning: pooled pretrain -> per-ticker head fine-tune.

Protocol (leakage-clean, same windowing/scaling discipline as Phase 2):
  * One expanding temporal split (train -> purge gap -> val) on the global dates.
  * Pretrain a TCN on the POOLED OTHER tickers' train windows (target excluded ->
    genuine transfer to a held-out ticker).
  * Fine-tune: freeze the pretrained encoder, train a fresh head on the TARGET's
    train windows.
  * Compare against a from-scratch TCN trained only on the target's train windows.
  * All three are scored on the SAME target val windows.

Scaling note: pretraining uses a scaler fit on the pool's train rows; the target
models use a scaler fit on the target's train rows. Per-feature standardization is
what lets the encoder transfer across tickers (every input is mean-0/std-1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score

from . import SEED
from .models.tcn import TCN, TrainConfig, WindowedFold, build_windowed_fold
from .splits import Fold


def single_temporal_fold(
    df: pd.DataFrame, train_frac: float = 0.8, purge_gap: int = 1
) -> Fold:
    """One expanding split over the dataset's unique dates with a purge gap."""
    uniq = pd.DatetimeIndex(pd.to_datetime(df["date"]).unique()).sort_values()
    cut = int(len(uniq) * train_frac)
    return Fold(
        index=0,
        train_start=uniq[0],
        train_end=uniq[cut - 1],
        val_start=uniq[cut - 1 + purge_gap + 1],
        val_end=uniq[-1],
    )


def _fit(
    model: TCN,
    wf: WindowedFold,
    cfg: TrainConfig,
    trainable: str = "all",
    seed: int = SEED,
) -> TCN:
    """Train `model` on wf with early stopping on val macro-F1.

    `trainable`: "all" optimizes every param; "head" freezes the encoder (`tcn`)
    and trains only `head` (the fine-tuning case).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if trainable == "head":
        for p in model.tcn.parameters():
            p.requires_grad_(False)
        params = model.head.parameters()
    else:
        params = model.parameters()

    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    counts = np.bincount(wf.ytr, minlength=3).astype(np.float64)
    weights = torch.tensor(counts.sum() / (3 * np.maximum(counts, 1)), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    Xtr = torch.from_numpy(wf.Xtr)
    ytr = torch.from_numpy(wf.ytr)
    Xva = torch.from_numpy(wf.Xva)
    n = len(Xtr)

    best_f1, best_state, bad = -1.0, None, 0
    for _ in range(cfg.max_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xva).argmax(1).numpy()
        f1 = f1_score(wf.yva, pred, labels=[0, 1, 2], average="macro", zero_division=0)
        if f1 > best_f1 + 1e-4:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@dataclass
class TransferResult:
    from_scratch_f1: float
    finetuned_f1: float
    pooled_zeroshot_f1: float
    from_scratch_recall: dict
    finetuned_recall: dict
    n_target_train: int
    n_target_val: int
    n_pool_train: int

    @property
    def improvement(self) -> float:
        return self.finetuned_f1 - self.from_scratch_f1

    def summary(self) -> str:
        def rc(d):
            return "/".join(f"{d[k]:.3f}" for k in ("low", "med", "high"))
        return (
            "=== Transfer learning (target=held-out ticker) ===\n"
            f"pool train windows: {self.n_pool_train} | "
            f"target train/val windows: {self.n_target_train}/{self.n_target_val}\n"
            f"  pooled zero-shot   macro-F1: {self.pooled_zeroshot_f1:.4f}\n"
            f"  from-scratch       macro-F1: {self.from_scratch_f1:.4f}  recall(l/m/h) {rc(self.from_scratch_recall)}\n"
            f"  pretrain+finetune  macro-F1: {self.finetuned_f1:.4f}  recall(l/m/h) {rc(self.finetuned_recall)}\n"
            f"  improvement (finetune - scratch): {self.improvement:+.4f}"
        )


def _recall_dict(y, pred) -> dict:
    r = recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {"low": float(r[0]), "med": float(r[1]), "high": float(r[2])}


def run_transfer(
    df: pd.DataFrame,
    target: str,
    cfg: TrainConfig | None = None,
    train_frac: float = 0.8,
) -> tuple[TransferResult, TCN]:
    """Run the full transfer experiment. Returns (result, finetuned_model)."""
    cfg = cfg or TrainConfig()
    fold = single_temporal_fold(df, train_frac=train_frac)

    pool_df = df[df["ticker"] != target].copy()
    target_df = df[df["ticker"] == target].copy()

    pool_wf = build_windowed_fold(pool_df, fold)
    target_wf = build_windowed_fold(target_df, fold)

    # 1) pretrain on pool
    pretrained = _fit(TCN(dropout=cfg.dropout), pool_wf, cfg, trainable="all")

    # 2) pooled zero-shot on target val (no target training at all)
    pretrained.eval()
    with torch.no_grad():
        zs_pred = pretrained(torch.from_numpy(target_wf.Xva)).argmax(1).numpy()
    zs_f1 = f1_score(target_wf.yva, zs_pred, labels=[0, 1, 2], average="macro", zero_division=0)

    # 3) from-scratch on target
    scratch = _fit(TCN(dropout=cfg.dropout), target_wf, cfg, trainable="all")
    scratch.eval()
    with torch.no_grad():
        sc_pred = scratch(torch.from_numpy(target_wf.Xva)).argmax(1).numpy()
    sc_f1 = f1_score(target_wf.yva, sc_pred, labels=[0, 1, 2], average="macro", zero_division=0)

    # 4) fine-tune head on target from the pretrained encoder
    ft = TCN(dropout=cfg.dropout)
    ft.load_state_dict({k: v.clone() for k, v in pretrained.state_dict().items()})
    # reinit head so it adapts to the target
    ft.head.reset_parameters()
    ft = _fit(ft, target_wf, cfg, trainable="head")
    ft.eval()
    with torch.no_grad():
        ft_pred = ft(torch.from_numpy(target_wf.Xva)).argmax(1).numpy()
    ft_f1 = f1_score(target_wf.yva, ft_pred, labels=[0, 1, 2], average="macro", zero_division=0)

    result = TransferResult(
        from_scratch_f1=float(sc_f1),
        finetuned_f1=float(ft_f1),
        pooled_zeroshot_f1=float(zs_f1),
        from_scratch_recall=_recall_dict(target_wf.yva, sc_pred),
        finetuned_recall=_recall_dict(target_wf.yva, ft_pred),
        n_target_train=len(target_wf.ytr),
        n_target_val=len(target_wf.yva),
        n_pool_train=len(pool_wf.ytr),
    )
    return result, ft
