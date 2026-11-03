"""ImageColdDiffusionGenerator restores its input and refuses unknown degradations (#2383)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from spectramr.models.generators.image_cold_diffusion_generator import (
    ImageColdDiffusionGenerator,
)


class _HalfDenoiser(nn.Module):
    """Stand-in for the U-Net: the prediction depends on the input, so the loop's start matters."""

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond=None) -> torch.Tensor:
        return 0.5 * x


def _generator(timesteps: int = 4) -> ImageColdDiffusionGenerator:
    gen = ImageColdDiffusionGenerator(
        in_channels=1,
        out_channels=1,
        timesteps=timesteps,
        degradation_type="noise",
        base_channels=32,
        channel_mults=(1,),
        num_res_blocks=1,
    ).eval()
    gen.denoising_model = _HalfDenoiser()
    return gen


def test_generate_depends_on_the_input() -> None:
    gen = _generator()
    x1, x2 = torch.rand(2, 1, 8, 8), torch.zeros(2, 1, 8, 8)
    torch.manual_seed(0)
    y1 = gen.generate(x1)
    torch.manual_seed(0)
    y2 = gen.generate(x2)
    assert not torch.allclose(y1, y2)


def test_generate_starts_the_reverse_loop_at_the_input() -> None:
    # The final step returns the denoiser's prediction, so with one timestep the
    # output is exactly 0.5 * input: the loop started from z, not from noise.
    gen = _generator(timesteps=1)
    z = torch.rand(2, 1, 8, 8)
    assert torch.equal(gen.generate(z), 0.5 * z)


def test_unknown_degradation_type_raises() -> None:
    with pytest.raises(ValueError, match="image_multi_physics"):
        ImageColdDiffusionGenerator(degradation_type="image_multi_physics")
