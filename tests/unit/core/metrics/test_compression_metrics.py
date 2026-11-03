"""``model_size_mb`` / ``compression_ratio`` resolve through the registry and count storage."""

import math

import pytest
import torch.nn as nn

from spectramr.core.metrics.compression_metrics import model_storage_bytes
from spectramr.core.metrics.registry import MetricsRegistry


def _net() -> nn.Module:
    return nn.Sequential(nn.Conv2d(1, 4, 3), nn.BatchNorm2d(4), nn.Linear(4, 2))


def test_storage_bytes_convention():
    net = _net()
    fp32 = model_storage_bytes(net, weight_bits=32)
    expected_fp32 = 4 * (
        sum(p.numel() for p in net.parameters()) + sum(b.numel() for b in net.buffers()) - 1
    )
    expected_fp32 += 8  # num_batches_tracked is int64
    assert fp32 == expected_fp32
    int8 = model_storage_bytes(net, weight_bits=8)
    # conv (36) and linear (8) weights drop from 4 to 1 byte each, plus a 4-byte scale per tensor.
    assert fp32 - int8 == 3 * (36 + 8) - 2 * 4


def test_registry_metrics():
    net = _net()
    size = MetricsRegistry.get("model_size")(model=net, weight_bits=8)
    assert size == pytest.approx(model_storage_bytes(net, weight_bits=8) / 2**20)
    ratio = MetricsRegistry.get("compression_ratio")(model=net, weight_bits=8)
    assert ratio == pytest.approx(
        model_storage_bytes(net, weight_bits=32) / model_storage_bytes(net, weight_bits=8)
    )
    assert MetricsRegistry.get("compression_ratio")(model=net) == pytest.approx(1.0)
    assert math.isnan(MetricsRegistry.get("model_size_mb")())


def test_unsupported_bits_raise():
    with pytest.raises(ValueError, match="weight_bits"):
        model_storage_bytes(_net(), weight_bits=2)
