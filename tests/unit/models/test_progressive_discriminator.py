"""Unit tests for the progressive-growing GAN discriminator."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.gans.progressive_discriminator import ProgressiveDiscriminator


class TestProgressiveDiscriminator:
    def test_constructible_with_no_args(self) -> None:
        disc = ProgressiveDiscriminator()
        assert disc.in_channels == 1
        assert disc.max_resolution == 256

    def test_rejects_non_power_of_two(self) -> None:
        with pytest.raises(ValueError):
            ProgressiveDiscriminator(max_resolution=48)

    def test_forward_phase0_scalar_score(self) -> None:
        disc = ProgressiveDiscriminator(max_resolution=64, base_channels=32)
        disc.set_phase(0, alpha=1.0)
        x = torch.randn(4, 1, 4, 4)
        out = disc(x)
        assert out.shape == (4, 1)

    def test_forward_each_phase(self) -> None:
        disc = ProgressiveDiscriminator(max_resolution=64, base_channels=32)
        for phase in range(disc.num_phases):
            disc.set_phase(phase, alpha=1.0)
            res = 4 * (2**phase)
            x = torch.randn(4, 1, res, res)
            out = disc(x)
            assert out.shape == (4, 1)
            assert torch.isfinite(out).all()

    def test_forward_fade_in_no_nan(self) -> None:
        disc = ProgressiveDiscriminator(max_resolution=64, base_channels=32)
        disc.set_phase(2, alpha=0.4)
        res = 4 * (2**2)
        x = torch.randn(4, 1, res, res)
        out = disc(x)
        assert torch.isfinite(out).all()

    def test_set_phase_validation(self) -> None:
        disc = ProgressiveDiscriminator(max_resolution=32)
        with pytest.raises(ValueError):
            disc.set_phase(99)
        with pytest.raises(ValueError):
            disc.set_phase(0, alpha=-0.1)

    def test_gradient_flow(self) -> None:
        disc = ProgressiveDiscriminator(max_resolution=32, base_channels=32)
        disc.set_phase(disc.num_phases - 1, alpha=1.0)
        res = disc.max_resolution
        x = torch.randn(4, 1, res, res, requires_grad=True)
        out = disc(x)
        out.mean().backward()
        assert x.grad is not None


def test_registered_as_discriminator_component() -> None:
    from spectramr.models.factories.model_factory import ModelFactory

    disc = ModelFactory().create_discriminator(
        model_type="progressive_discriminator", in_channels=1, max_resolution=16, base_channels=8
    )
    assert isinstance(disc, ProgressiveDiscriminator)
    assert callable(disc.set_phase)
