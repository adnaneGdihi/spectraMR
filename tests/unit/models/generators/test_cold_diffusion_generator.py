"""``ColdDiffusionGenerator`` hands the denoiser its timestep and ``cond`` (#2020).

Paired with ``src/spectramr/models/generators/cold_diffusion_generator.py``.
"""

from __future__ import annotations

import logging

import pytest
import torch
from torch import nn

from spectramr.models.generators.cold_diffusion_generator import ColdDiffusionGenerator


class _TimeCondDenoiser(nn.Module):
    """Records each call; output depends on ``cond`` so guidance is measurable."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.calls: list[tuple[torch.Tensor | None, torch.Tensor | None]] = []

    def forward(self, x, timesteps=None, cond=None):
        self.calls.append((timesteps, cond))
        out = x * self.scale
        return out if cond is None else out + cond


class _TimeBlindDenoiser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x * self.scale


def _gen(denoiser: nn.Module) -> ColdDiffusionGenerator:
    return ColdDiffusionGenerator(denoising_model=denoiser, timesteps=10)


def test_training_forward_passes_the_sampled_timestep():
    denoiser = _TimeCondDenoiser()
    _gen(denoiser)(torch.randn(3, 1, 8, 8))
    ((t, cond),) = denoiser.calls
    assert t is not None and t.shape == (3,)
    assert cond is None


def test_caller_timestep_skips_the_internal_noising():
    """A bridge/interpolant strategy forms x_t itself and passes a continuous t."""
    denoiser = _TimeCondDenoiser()
    x_t = torch.randn(2, 1, 8, 8)
    t = torch.rand(2)
    out = _gen(denoiser)(x_t, t)
    assert torch.equal(out, x_t)
    assert denoiser.calls[0][0] is t


def test_guidance_term_is_nonzero_with_cond():
    denoiser = _TimeCondDenoiser()
    gen = _gen(denoiser).eval()
    x = torch.zeros(1, 1, 4, 4)
    cond = torch.ones(1, 1, 4, 4)
    t = torch.full((1,), 5, dtype=torch.long)
    out = gen.p_sample(x, t, 0, cond=cond, guidance_scale=2.0)
    # uncond = 0, cond = 1 -> 0 + 2 * (1 - 0)
    assert torch.allclose(out, torch.full_like(out, 2.0))
    assert [c is cond for _, c in denoiser.calls] == [False, True]


def test_cond_on_a_cond_blind_denoiser_raises():
    gen = _gen(_TimeBlindDenoiser()).eval()
    with pytest.raises(ValueError, match="ignores `cond`"):
        gen.p_sample(
            torch.zeros(1, 1, 4, 4),
            torch.zeros(1, dtype=torch.long),
            0,
            cond=torch.ones(1, 1, 4, 4),
            guidance_scale=2.0,
        )


def test_time_blind_denoiser_is_reported(caplog):
    with caplog.at_level(logging.WARNING):
        gen = _gen(_TimeBlindDenoiser())
    assert "ignores the timestep" in caplog.text
    assert gen(torch.randn(1, 1, 4, 4)).shape == (1, 1, 4, 4)


def _tiny_unet(**kwargs):
    from spectramr.models.reconstruction.unet import UNet

    return UNet(in_channels=1, out_channels=1, **kwargs)


def test_guidance_with_the_production_unet_raises():
    """``enhanced_deep_unet`` builds this UNet; it drops ``cond``, so guidance would be zero."""
    gen = _gen(_tiny_unet(model_channels=8, channel_mult=(1, 2))).eval()
    with pytest.raises(ValueError, match="ignores `cond`"):
        gen.p_sample(
            torch.zeros(1, 1, 8, 8),
            torch.full((1,), 5, dtype=torch.long),
            0,
            cond=torch.ones(1, 1, 8, 8),
            guidance_scale=2.0,
        )


def test_production_unet_without_time_embedding_is_reported(caplog):
    with caplog.at_level(logging.WARNING):
        _gen(_tiny_unet(depth=1))
    assert "ignores the timestep" in caplog.text


def test_production_unet_with_time_embedding_receives_the_timestep(caplog):
    unet = _tiny_unet(model_channels=8, channel_mult=(1, 2))
    seen = []
    unet.time_mlp.register_forward_hook(lambda _m, inp, _o: seen.append(inp[0]))
    with caplog.at_level(logging.WARNING):
        gen = _gen(unet).eval()
    assert "ignores the timestep" not in caplog.text
    with torch.no_grad():
        gen(torch.randn(2, 1, 8, 8))
    assert len(seen) == 1 and seen[0].shape == (2,)
