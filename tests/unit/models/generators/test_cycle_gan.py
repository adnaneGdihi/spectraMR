"""Unit tests for the CycleGAN ResNet generator's Z_anat tap."""

import torch

from spectramr.models.generators.cycle_gan import ResNetGenerator


def test_forward_with_anatomy_matches_forward_and_exposes_bottleneck() -> None:
    torch.manual_seed(0)
    gen = ResNetGenerator(1, 1, ngf=8, n_blocks=2).eval()
    x = torch.rand(2, 1, 32, 32)
    out, z = gen.forward_with_anatomy(x)
    assert torch.allclose(out, gen(x))
    assert gen.latent_channels == 32
    assert z.shape == (2, 32, 8, 8)


def test_anatomy_tap_keeps_state_dict_keys() -> None:
    gen = ResNetGenerator(1, 1, ngf=8, n_blocks=2)
    assert all(k.startswith("model.") for k in gen.state_dict())
