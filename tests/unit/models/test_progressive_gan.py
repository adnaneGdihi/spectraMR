"""Unit tests for the progressive-growing GAN generator and its blocks."""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.models.blocks.progressive import FadeIn, MinibatchStdDev, PixelNorm
from spectramr.models.gans.progressive_gan import ProgressiveGAN
from spectramr.models.registry import MODEL_REGISTRY, get_model_class


class TestProgressiveBlocks:
    def test_pixel_norm_unit_rms(self) -> None:
        pn = PixelNorm()
        x = torch.randn(2, 8, 4, 4) * 10.0
        out = pn(x)
        # Per-pixel RMS over channels should be ~1.
        rms = out.pow(2).mean(dim=1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)

    def test_pixel_norm_rejects_bad_epsilon(self) -> None:
        with pytest.raises(ValueError):
            PixelNorm(epsilon=0.0)

    def test_minibatch_stddev_adds_one_channel(self) -> None:
        mb = MinibatchStdDev(group_size=2)
        x = torch.randn(4, 8, 4, 4)
        out = mb(x)
        assert out.shape == (4, 9, 4, 4)

    def test_minibatch_stddev_small_batch(self) -> None:
        mb = MinibatchStdDev(group_size=4)
        x = torch.randn(2, 3, 8, 8)  # batch smaller than group_size
        out = mb(x)
        assert out.shape == (2, 4, 8, 8)

    def test_fade_in_endpoints(self) -> None:
        fade = FadeIn()
        old = torch.zeros(1, 1, 4, 4)
        new = torch.ones(1, 1, 4, 4)
        assert torch.allclose(fade(old, new, 0.0), old)
        assert torch.allclose(fade(old, new, 1.0), new)
        assert torch.allclose(fade(old, new, 0.5), 0.5 * torch.ones_like(new))

    def test_fade_in_rejects_bad_alpha(self) -> None:
        fade = FadeIn()
        with pytest.raises(ValueError):
            fade(torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2), 1.5)

    def test_fade_in_rejects_shape_mismatch(self) -> None:
        fade = FadeIn()
        with pytest.raises(ValueError):
            fade(torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 4, 4), 0.5)


class TestProgressiveGAN:
    def test_registered(self) -> None:
        assert "progressive_gan" in MODEL_REGISTRY
        assert get_model_class("progressive_gan") is ProgressiveGAN
        assert MODEL_REGISTRY["progressive_gan"]["mode"] == "gan"

    def test_constructible_with_no_args(self) -> None:
        model = ProgressiveGAN()
        assert model.z_dim == 512
        assert model.max_resolution == 256

    def test_num_phases(self) -> None:
        model = ProgressiveGAN(max_resolution=64)
        assert model.num_phases == int(math.log2(64)) - 1  # == 5

    def test_rejects_non_power_of_two(self) -> None:
        with pytest.raises(ValueError):
            ProgressiveGAN(max_resolution=100)

    def test_forward_latent_phase0(self) -> None:
        model = ProgressiveGAN(z_dim=16, max_resolution=64, base_channels=32)
        model.set_phase(0, alpha=1.0)
        z = torch.randn(2, 16)
        out = model(z)
        assert out.shape == (2, 1, 4, 4)

    def test_resolution_doubles_per_phase(self) -> None:
        model = ProgressiveGAN(z_dim=16, max_resolution=64, base_channels=32)
        z = torch.randn(2, 16)
        for phase in range(model.num_phases):
            model.set_phase(phase, alpha=1.0)
            out = model(z)
            expected = 4 * (2**phase)
            assert out.shape == (2, 1, expected, expected)

    def test_forward_fade_in_no_nan(self) -> None:
        model = ProgressiveGAN(z_dim=16, max_resolution=64, base_channels=32)
        z = torch.randn(2, 16)
        model.set_phase(2, alpha=0.3)
        out = model(z)
        assert torch.isfinite(out).all()
        assert out.shape == (2, 1, 16, 16)

    def test_forward_image_conditioning(self) -> None:
        model = ProgressiveGAN(z_dim=16, max_resolution=32, base_channels=32)
        model.set_phase(model.num_phases - 1, alpha=1.0)
        img = torch.randn(2, 1, 32, 32)
        out = model(img)
        assert out.shape[0] == 2 and out.shape[1] == 1

    def test_set_phase_validation(self) -> None:
        model = ProgressiveGAN(max_resolution=32)
        with pytest.raises(ValueError):
            model.set_phase(model.num_phases + 5)
        with pytest.raises(ValueError):
            model.set_phase(0, alpha=2.0)

    def test_gradient_flow(self) -> None:
        model = ProgressiveGAN(z_dim=16, max_resolution=32, base_channels=32)
        model.set_phase(1, alpha=0.5)
        z = torch.randn(2, 16)
        out = model(z)
        out.mean().backward()
        assert model.project.weight.grad is not None

    def test_image_conditioning_sees_spatial_layout(self) -> None:
        # #2384: a pixel permutation keeps the mean but must change the output.
        torch.manual_seed(0)
        model = ProgressiveGAN(z_dim=16, max_resolution=16, base_channels=8).eval()
        x = torch.rand(2, 1, 16, 16)
        x2 = x[:, :, torch.randperm(16)][:, :, :, torch.randperm(16)]
        with torch.no_grad():
            assert (model(x) - model(x2)).abs().max() > 1e-4

    def test_conditioning_params_exist_before_first_forward(self) -> None:
        model = ProgressiveGAN(z_dim=16, max_resolution=16, base_channels=8)
        opt_ids = {id(p) for p in model.parameters()}
        assert all(id(p) in opt_ids for p in model.cond_proj.parameters())
        n_before = sum(1 for _ in model.parameters())
        model(torch.rand(1, 1, 16, 16))
        assert sum(1 for _ in model.parameters()) == n_before

    def test_conditioning_channel_mismatch_raises(self) -> None:
        model = ProgressiveGAN(in_channels=1, z_dim=16, max_resolution=16, base_channels=8)
        with pytest.raises(ValueError, match="in_channels"):
            model(torch.rand(1, 2, 16, 16))
