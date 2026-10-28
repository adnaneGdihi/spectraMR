import pytest
import torch

from spectramr.models.layers.layer_norm_2d import LayerNorm2d

pytestmark = pytest.mark.unit


def test_normalizes_channel_axis_and_preserves_shape() -> None:
    x = torch.randn(2, 6, 5, 4)
    y = LayerNorm2d(6)(x)
    assert y.shape == x.shape
    assert torch.allclose(y.mean(1), torch.zeros(2, 5, 4), atol=1e-5)


def test_matches_timm_layer_norm_2d() -> None:
    try:
        from timm.layers import LayerNorm2d as TimmLayerNorm2d
    except ImportError:
        pytest.skip("timm not installed")
    x = torch.randn(2, 6, 5, 4)
    ours, ref = LayerNorm2d(6), TimmLayerNorm2d(6)
    with torch.no_grad():
        for m in (ours, ref):
            m.weight.copy_(torch.linspace(0.5, 1.5, 6))
            m.bias.copy_(torch.linspace(-1, 1, 6))
    assert ours.eps == ref.eps
    assert torch.allclose(ours(x), ref(x), atol=1e-6)
