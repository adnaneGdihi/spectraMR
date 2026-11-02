"""Pruning and weight quantization in ``models/model_compression.py`` do what they declare."""

import pytest
import torch
import torch.nn as nn

from spectramr.models.model_compression import (
    CompressionConfig,
    ModelCompressionPipeline,
    PruningManager,
    QuantizationManager,
    quantize_tensor_symmetric,
)


def _net() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(1, 8, 3), nn.ReLU(), nn.ConvTranspose2d(8, 4, 2), nn.Linear(4, 3)
    )


def _weights(net):
    return [
        m.weight for m in net.modules() if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear))
    ]


def test_symmetric_quantization_and_zero_tensor():
    levels, scale = quantize_tensor_symmetric(torch.tensor([-1.0, 0.5, 1.0]), 8)
    assert levels.dtype == torch.int8 and levels.tolist() == [-127, 64, 127]
    levels, scale = quantize_tensor_symmetric(torch.zeros(4), 4)
    assert torch.all(levels == 0) and scale.item() == 1.0
    with pytest.raises(ValueError, match="weight_bits"):
        quantize_tensor_symmetric(torch.ones(2), 3)


def test_weights_land_on_the_int8_grid_and_biases_stay():
    net = _net()
    q = QuantizationManager(CompressionConfig(weight_bits=8)).quantize_model(net)
    for w in _weights(q):
        scale = w.abs().max() / 127
        assert torch.allclose(w / scale, torch.round(w / scale), atol=1e-4)
    assert torch.equal(q[0].bias, net[0].bias)


def test_activation_quantization_raises():
    mgr = QuantizationManager(CompressionConfig(quantize_activations=True))
    with pytest.raises(NotImplementedError):
        mgr.quantize_model(_net(), calibration_data=torch.zeros(1))


@pytest.mark.parametrize("method", ["l1_global", "random"])
def test_unstructured_pruning_zeroes_the_declared_fraction(method):
    pruned = PruningManager(
        CompressionConfig(pruning_method=method, pruning_ratio=0.5)
    ).prune_model(_net())
    flat = torch.cat([w.flatten() for w in _weights(pruned)])
    assert (flat == 0).float().mean().item() == pytest.approx(0.5, abs=0.12)


def test_l2_filter_pruning_zeroes_whole_filters():
    pruned = PruningManager(
        CompressionConfig(pruning_method="l2_filter", pruning_ratio=0.5)
    ).prune_model(_net())
    conv = pruned[0].weight
    zero_filters = (conv.flatten(1).abs().sum(1) == 0).sum().item()
    assert zero_filters == 4


def test_unknown_pruning_method_raises():
    with pytest.raises(ValueError, match="pruning_method"):
        PruningManager(CompressionConfig(pruning_method="structured")).prune_model(_net())


def test_pipeline_stats_use_the_storage_convention():
    pipe = ModelCompressionPipeline(
        CompressionConfig(weight_bits=8, pruning_method="l1_global", pruning_ratio=0.3)
    )
    out = pipe.compress_model(_net())
    assert isinstance(out, nn.Module)
    assert 3.0 < pipe.get_compression_stats().compression_ratio < 4.0


def test_l2_filter_pruning_uses_the_output_dim_of_transposed_convs():
    pruned = PruningManager(
        CompressionConfig(pruning_method="l2_filter", pruning_ratio=0.5)
    ).prune_model(_net())
    tconv = pruned[2].weight  # (in=8, out=4, 2, 2)
    assert (tconv.transpose(0, 1).flatten(1).abs().sum(1) == 0).sum().item() == 2
