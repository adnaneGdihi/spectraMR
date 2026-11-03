"""monotone_remap_invariance: the remap is monotone, MCGI reads 0, the rank-off twin does not."""

from __future__ import annotations

import pytest
import torch

from spectramr.core.metrics.context import MetricContext
from spectramr.core.metrics.monotone_remap_invariance import (
    MonotoneRemapInvariance,
    random_monotone_remap,
)
from spectramr.core.metrics.registry import MetricsRegistry
from spectramr.models.encoders.mcgi_encoder import MCGIEncoder

pytestmark = pytest.mark.unit


def _encoder(rank_transform: bool) -> MCGIEncoder:
    torch.manual_seed(0)
    return MCGIEncoder(hidden_channels=4, depth=2, rank_transform=rank_transform).eval()


def test_the_remap_preserves_order_and_ties_and_changes_values() -> None:
    x = torch.rand(3, 1, 16, 16)
    x[:, :, :4] = 0.0  # a tied background
    y = random_monotone_remap(x, torch.Generator().manual_seed(1))
    assert y.shape == x.shape and y.dtype == x.dtype
    assert not torch.allclose(x, y)
    for b in range(3):
        xs, ys = x[b].flatten(), y[b].flatten()
        order = xs.argsort()
        assert bool((ys[order].diff() >= 0).all())
        assert torch.equal(xs.unsqueeze(0) == xs.unsqueeze(1), ys.unsqueeze(0) == ys.unsqueeze(1))


def test_a_constant_image_stays_constant_and_half_precision_raises() -> None:
    x = torch.full((1, 1, 4, 4), 3.0)
    y = random_monotone_remap(x, torch.Generator().manual_seed(0))
    assert torch.equal(y, y.flatten()[0].expand_as(y))
    with pytest.raises(ValueError, match="float32"):
        random_monotone_remap(x.half(), torch.Generator())


def test_full_size_images_stay_exact() -> None:
    """192x192 uniform noise: a float-evaluated map merged neighbours here (2.98e-08)."""
    torch.manual_seed(3)
    x = torch.rand(2, 1, 192, 192)
    metric = MonotoneRemapInvariance()
    score = metric(x, context=MetricContext(reconstructor=_encoder(True), model_input=x))
    assert float(score) == 0.0


def test_registered_lower_is_better_with_its_needs() -> None:
    assert MetricsRegistry.needs("monotone_remap_invariance") == ("reconstructor", "model_input")
    assert MonotoneRemapInvariance.higher_is_better is False


def test_rank_on_reads_zero_and_rank_off_does_not() -> None:
    x = torch.rand(2, 1, 16, 16)
    metric = MonotoneRemapInvariance(num_remaps=3)
    on = metric(x, context=MetricContext(reconstructor=_encoder(True), model_input=x))
    off = metric(x, context=MetricContext(reconstructor=_encoder(False), model_input=x))
    assert float(on) == 0.0
    assert float(off) > 1e-3


def test_missing_context_and_complex_input_raise() -> None:
    x = torch.rand(1, 1, 8, 8)
    with pytest.raises(ValueError, match="model_input"):
        MonotoneRemapInvariance()(x, context=MetricContext(reconstructor=lambda t: t))
    z = torch.complex(x, x)
    with pytest.raises(ValueError, match="complex"):
        MonotoneRemapInvariance()(
            z, context=MetricContext(reconstructor=lambda t: t, model_input=z)
        )
    with pytest.raises(ValueError, match="num_remaps"):
        MonotoneRemapInvariance(num_remaps=0)
