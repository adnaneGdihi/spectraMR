"""Tests for the LM-13 latent-regression control."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.latent_diffusion_generator import LatentDiffusionGenerator
from spectramr.models.generators.latent_regression_generator import LatentRegressionGenerator

_KW = {
    "latent_channels": 4,
    "base_channels": 16,
    "timesteps": 5,
    "spatial_dims": 2,
    "num_layers": 3,
    "conditional_translation": True,
    "use_contrast_guidance": True,
    "num_contrasts": 4,
}


def _control(**overrides) -> LatentRegressionGenerator:
    return LatentRegressionGenerator(in_channels=1, out_channels=1, **{**_KW, **overrides})


def test_it_is_registered():
    from spectramr.models.registry import get_model_class

    assert get_model_class("latent_regression") is LatentRegressionGenerator


def test_forward_returns_an_image_of_the_input_size():
    out = _control()(torch.randn(2, 1, 32, 32), contrast_id=torch.tensor([0, 3]))
    assert out.shape == (2, 1, 32, 32)


def test_it_has_exactly_the_ldm_architecture():
    """Single knob vs the LDM: same parameters and buffers, by name and shape."""
    control = {k[len("ldm.") :]: v.shape for k, v in _control().state_dict().items()}
    ldm = {
        k: v.shape
        for k, v in LatentDiffusionGenerator(in_channels=1, out_channels=1, **_KW)
        .state_dict()
        .items()
    }
    assert control == ldm


def test_a_missing_contrast_raises():
    with pytest.raises(ValueError, match="contrast_id is None"):
        _control()(torch.randn(1, 1, 32, 32))


def test_it_needs_the_ulf_condition():
    with pytest.raises(ValueError, match="conditional_translation"):
        _control(conditional_translation=False)


def test_the_contrast_reaches_the_output():
    torch.manual_seed(0)
    control = _control().eval()
    with torch.no_grad():
        for p in control.ldm.denoise_net.parameters():
            if not p.any():
                p.normal_(0.0, 0.1)
    x = torch.randn(1, 1, 32, 32).expand(2, 1, 32, 32)
    out = control(x, contrast_id=torch.tensor([0, 2]))
    assert not torch.allclose(out[0], out[1])


def test_the_stage1_load_freezes_the_vae_and_trains_the_unet():
    control = _control()
    stage1 = {f"autoencoder.{k}": v for k, v in control.ldm.autoencoder.state_dict().items()}
    control.load_autoencoder_state(stage1, freeze=True, source="stage1")
    control.train()
    control(torch.randn(2, 1, 32, 32), contrast_id=torch.tensor([0, 1])).pow(2).mean().backward()
    assert all(p.grad is None for p in control.ldm.autoencoder.parameters())
    assert any(p.grad is not None for p in control.ldm.denoise_net.parameters())


def test_the_builder_refreeze_path_accepts_its_own_state_dict():
    """ModelBuilder._freeze_stage1_vae passes generator.state_dict(): keys start ldm."""
    control = _control()
    control.load_autoencoder_state(control.state_dict(), freeze=True, source="self")
    assert all(not p.requires_grad for p in control.ldm.autoencoder.parameters())
