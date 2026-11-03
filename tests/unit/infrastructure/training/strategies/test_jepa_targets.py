"""Block pooling, block geometry and the encoder representation the JEPA objective reads."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.training.strategies.jepa_targets import (
    GEOMETRY_DIM,
    block_geometry,
    encoder_features,
    flatten_volume_rows,
    pool_over_mask,
)
from spectramr.models.interfaces.latent_access import LatentAccessMixin


class _LatentNet(LatentAccessMixin, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = torch.nn.Conv2d(1, 6, 3, stride=2, padding=1)
        self.dec = torch.nn.Conv2d(6, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.enc(x)
        self._set_latent_state(embedding=z.mean((-2, -1)), bottleneck=z)
        return torch.nn.functional.interpolate(self.dec(z), size=x.shape[-2:])


def test_a_latent_model_is_read_at_its_bottleneck_not_its_output() -> None:
    net = _LatentNet()
    out = net(torch.randn(2, 1, 8, 8))
    feats = encoder_features(net, out)
    assert feats.shape == (2, 6, 4, 4)


def test_a_plain_encoder_is_read_at_its_output_and_a_vector_raises() -> None:
    conv = torch.nn.Conv2d(1, 3, 3, padding=1)
    out = conv(torch.randn(1, 1, 8, 8))
    assert encoder_features(conv, out) is out
    with pytest.raises(ValueError, match="encoder feature map"):
        encoder_features(torch.nn.Linear(4, 4), torch.randn(2, 4))


def test_pool_over_mask_averages_the_covered_cells() -> None:
    feats = torch.arange(16.0).view(1, 1, 4, 4)
    mask = torch.zeros(8, 8, dtype=torch.bool)
    mask[:4, :4] = True  # the top-left 2x2 feature cells
    assert pool_over_mask(feats, mask).item() == pytest.approx((0 + 1 + 4 + 5) / 4)


def test_block_geometry_reads_centre_and_extent() -> None:
    mask = torch.zeros(10, 20, dtype=torch.bool)
    mask[2:4, 10:20] = True
    geom = block_geometry(mask)
    assert geom.shape == (GEOMETRY_DIM,)
    assert torch.allclose(geom, torch.tensor([0.3, 0.75, 0.2, 0.5]))


def test_flatten_volume_rows_lays_slices_sample_after_sample() -> None:
    vol = torch.arange(2 * 3).float().view(2, 1, 3, 1, 1)
    assert flatten_volume_rows(vol).flatten().tolist() == [0, 1, 2, 3, 4, 5]
