"""Phase-3 quantization tests (offline, random weights)."""

from __future__ import annotations

import numpy as np
import torch

from stockvol.models.tcn import N_FEATURES, TCN, WINDOW
from stockvol.quantize import (
    QuantizableTCN,
    dynamic_quantize,
    model_size_bytes,
    static_quantize,
    to_quantizable,
)


def test_quantizable_loads_fp32_weights():
    m = TCN().eval()
    q = to_quantizable(m)
    # shared param names carry over identically
    assert torch.allclose(q.head.weight, m.head.weight)
    assert isinstance(q, QuantizableTCN)


def test_static_quant_shrinks_and_runs():
    m = TCN().eval()
    X = np.random.randn(200, N_FEATURES, WINDOW).astype(np.float32)
    s = static_quantize(m, X[:100])
    # conv-heavy net -> meaningful shrink
    assert model_size_bytes(s) < 0.6 * model_size_bytes(m)
    # quantized model still produces (N,3) logits
    with torch.no_grad():
        out = s(torch.from_numpy(X[:8]))
    assert out.shape == (8, 3)


def test_dynamic_quant_is_noop_on_conv_net():
    """Documents the honest finding: dynamic INT8 barely changes a conv TCN."""
    m = TCN().eval()
    d = dynamic_quantize(m)
    ratio = model_size_bytes(d) / model_size_bytes(m)
    assert 0.95 < ratio < 1.05  # ~unchanged (only the tiny Linear head is eligible)
