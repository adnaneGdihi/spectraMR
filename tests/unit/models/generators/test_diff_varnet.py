"""Unit tests for :mod:`spectramr.models.generators.diff_varnet`."""

from __future__ import annotations

from unittest import mock

import pytest
import torch

import spectramr.models.generators.diff_varnet as mod
from spectramr.models.generators.diff_varnet import DiffVarNet

pytestmark = pytest.mark.unit


def _tiny() -> DiffVarNet:
    torch.manual_seed(0)
    return DiffVarNet(
        in_channels=2,
        out_channels=2,
        image_size=32,
        num_unrolls=1,
        base_channels=8,
        physics_emb_dim=32,
    ).eval()


def _scales_seen(**forward_kwargs) -> list[str]:
    seen: list[str] = []
    real = mod.sinusoidal_timestep_embedding

    def spy(t, dim, *, max_timesteps=None, scale="unit_interval"):
        seen.append(scale)
        return real(t, dim, max_timesteps=max_timesteps, scale=scale)

    with mock.patch.object(mod, "sinusoidal_timestep_embedding", spy), torch.no_grad():
        _tiny()(torch.randn(1, 2, 32, 32), torch.tensor([3]), max_timesteps=29.0, **forward_kwargs)
    return seen


@pytest.mark.parametrize("scale", ["unit_interval", "integer"])
def test_timestep_scale_reaches_the_sinusoid(scale: str) -> None:
    """A dropped ``timestep_scale`` keeps the dividing form whatever the arm declared (#2315)."""
    assert _scales_seen(timestep_scale=scale) == [scale]


def test_acceleration_conditioning_is_never_rescaled() -> None:
    """An acceleration factor is not a step index; it keeps the legacy path."""
    seen = _scales_seen(timestep_scale="integer", acceleration=torch.tensor([4.0]))
    assert seen == ["unit_interval"]


class TestContrastEmbeddingBatch:
    """Exactly one contrast row per sample; a truncated batch or a broadcast row raises."""

    @staticmethod
    def _net() -> DiffVarNet:
        torch.manual_seed(0)
        return DiffVarNet(in_channels=2, out_channels=2, image_size=32, base_channels=8).eval()

    def test_contrast_batch_larger_than_the_timestep_batch_raises(self) -> None:
        net = self._net()
        with pytest.raises(ValueError, match="contrast_emb batch 3"), torch.no_grad():
            net(torch.randn(2, 2, 32, 32), torch.tensor([1, 2]), contrast_emb=torch.randn(3, 256))

    @pytest.mark.parametrize(("rows", "t_rows"), [(1, 2), (2, 3)])
    def test_a_broadcast_row_or_a_larger_timestep_batch_raises(self, rows, t_rows) -> None:
        """A single row used to broadcast onto every sample (#2055)."""
        net = self._net()
        x = torch.randn(t_rows, 2, 32, 32)
        with pytest.raises(ValueError, match=f"contrast_emb batch {rows}"), torch.no_grad():
            net(x, torch.arange(1, t_rows + 1), contrast_emb=torch.randn(rows, 256))

    @pytest.mark.parametrize("rows", [2])
    def test_per_sample_rows_pass(self, rows) -> None:
        net = self._net()
        x = torch.randn(2, 2, 32, 32)
        with torch.no_grad():
            out = net(x, torch.tensor([1, 2]), contrast_emb=torch.randn(rows, 256))
        assert out.shape == x.shape

    def test_a_per_volume_contrast_is_expanded_over_the_slices(self) -> None:
        """A ``[B, C, D, H, W]`` input flattens to ``B*D`` rows; B contrast rows must follow it."""
        net = self._net()
        x = torch.randn(2, 2, 3, 32, 32)
        with torch.no_grad():
            out = net(x, torch.tensor([1, 2]), contrast_emb=torch.randn(2, 256))
        assert out.shape == x.shape
