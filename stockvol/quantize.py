"""Phase 3b — INT8 post-training quantization + size/latency/accuracy table.

Two methods (torch.ao.quantization, eager mode):
  * DYNAMIC INT8 (`quantize_dynamic`, {Linear}) — what the brief asks for. On THIS
    model it is a near no-op: the TCN is ~99% Conv1d params and dynamic quant only
    touches Linear/RNN. We report it honestly to make the point.
  * STATIC INT8 PTQ — inserts observers, calibrates on real windows, and converts
    Conv1d -> quantized INT8. This is the method that actually shrinks/speeds a
    conv net, so it carries the real tradeoff.

`QuantizableTCN` mirrors `TCN`'s submodule names so it can load trained FP32
weights, adding QuantStub/DeQuantStub and a FloatFunctional for the residual add.
"""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from .models.tcn import TCN, Chomp1d, N_FEATURES


# --- quantizable mirror of the TCN ----------------------------------------
class QTemporalBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel, padding=pad, dilation=dilation),
            Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(c_out, c_out, kernel, padding=pad, dilation=dilation),
            Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None
        self.relu = nn.ReLU()
        self.add = nn.quantized.FloatFunctional()  # quant-aware residual add

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(self.add.add(out, res))


class QuantizableTCN(nn.Module):
    def __init__(
        self,
        n_features: int = N_FEATURES,
        channels: tuple[int, ...] = (32, 32, 64),
        kernel: int = 3,
        dropout: float = 0.2,
        n_classes: int = 3,
    ):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        layers = []
        c_in = n_features
        for i, c_out in enumerate(channels):
            layers.append(QTemporalBlock(c_in, c_out, kernel, 2**i, dropout))
            c_in = c_out
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(c_in, n_classes)  # kept in float (tiny)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        h = self.tcn(x)
        h = h[:, :, -1]
        h = self.dequant(h)
        return self.head(h)


def to_quantizable(model: TCN) -> QuantizableTCN:
    """Copy trained FP32 weights into the quantizable architecture."""
    q = QuantizableTCN()
    q.load_state_dict(model.state_dict(), strict=False)  # quant/dequant/add have no params
    q.eval()
    return q


def static_quantize(model: TCN, calib_X: np.ndarray, backend: str = "x86") -> nn.Module:
    """Static INT8 PTQ: observe -> calibrate on `calib_X` -> convert to INT8."""
    torch.backends.quantized.engine = backend
    q = to_quantizable(model)
    q.qconfig = torch.ao.quantization.get_default_qconfig(backend)
    # Keep the head in FP32: we dequant before it, so it must stay float.
    q.head.qconfig = None
    torch.ao.quantization.prepare(q, inplace=True)
    with torch.no_grad():  # calibration pass (collect activation ranges)
        for i in range(0, len(calib_X), 512):
            q(torch.from_numpy(calib_X[i : i + 512]))
    torch.ao.quantization.convert(q, inplace=True)
    return q


def dynamic_quantize(model: TCN) -> nn.Module:
    """Dynamic INT8 (Linear only) — as asked; near no-op on this conv net."""
    return torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(model).eval(), {nn.Linear}, dtype=torch.qint8
    )


# --- measurement -----------------------------------------------------------
def model_size_bytes(model: nn.Module) -> int:
    path = "_qsize.pt"
    torch.save(model.state_dict(), path)
    n = os.path.getsize(path)
    os.remove(path)
    return n


def latency_ms_per_sample(model: nn.Module, X: np.ndarray, batch: int = 256, reps: int = 5) -> float:
    model.eval()
    xb = torch.from_numpy(X[:batch])
    with torch.no_grad():
        for _ in range(2):  # warmup
            model(xb)
        t0 = time.perf_counter()
        for _ in range(reps):
            model(xb)
        dt = (time.perf_counter() - t0) / reps
    return 1000.0 * dt / batch


def macro_f1(model: nn.Module, X: np.ndarray, y: np.ndarray) -> float:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            preds.append(model(torch.from_numpy(X[i : i + 512])).argmax(1).numpy())
    pred = np.concatenate(preds)
    return float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0))


@dataclass
class QuantRow:
    name: str
    size_bytes: int
    latency_ms: float
    macro_f1: float


def benchmark(
    fp32: TCN, calib_X: np.ndarray, val_X: np.ndarray, val_y: np.ndarray
) -> list[QuantRow]:
    """Build the FP32 / dynamic-INT8 / static-INT8 comparison rows."""
    fp32 = fp32.eval()
    dyn = dynamic_quantize(fp32)
    stat = static_quantize(fp32, calib_X)
    rows = []
    for name, m in [("FP32", fp32), ("dynamic-INT8", dyn), ("static-INT8", stat)]:
        rows.append(
            QuantRow(name, model_size_bytes(m),
                     latency_ms_per_sample(m, val_X), macro_f1(m, val_X, val_y))
        )
    return rows


def format_table(rows: list[QuantRow]) -> str:
    base = rows[0]
    out = [
        f"{'model':>14} {'size(KB)':>9} {'size_x':>7} {'lat(ms/smp)':>12} {'speedup':>8} {'macroF1':>8} {'ΔF1':>8}",
        "-" * 74,
    ]
    for r in rows:
        out.append(
            f"{r.name:>14} {r.size_bytes/1024:>9.1f} "
            f"{base.size_bytes/r.size_bytes:>6.2f}x {r.latency_ms:>12.4f} "
            f"{base.latency_ms/r.latency_ms:>7.2f}x {r.macro_f1:>8.4f} "
            f"{r.macro_f1-base.macro_f1:>+8.4f}"
        )
    return "\n".join(out)
