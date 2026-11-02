"""Tests for the latent-diffusion + pixel-space residual-head wrapper.

Multi-contrast Item 2 (``TODO/backlog_multi_contrast_remaining.md``).
The wrapper composes ``LatentDiffusionGenerator`` (backbone) with a
small UNet residual head that adds back the high-frequency content the
autoencoder bottleneck discarded.
"""
from __future__ import annotations

import pytest
import torch


def test_wrapper_is_registered() -> None:
    """The model must register under its canonical key."""
    import spectramr.models.generators  # noqa: F401
    from spectramr.models.registry import list_models

    assert "latent_diffusion_residual_head" in list_models()


def test_wrapper_supports_contrast_conditioning_capability() -> None:
    """Registry must reflect ``supports_contrast_conditioning=True``."""
    import spectramr.models.generators  # noqa: F401
    from spectramr.models.registry import model_supports

    assert model_supports(
        "latent_diffusion_residual_head", "supports_contrast_conditioning"
    )


def test_residual_head_param_ratio_under_10_percent() -> None:
    """Acceptance criterion: head must stay lightweight (< 10 % of backbone)."""
    from spectramr.models.generators.latent_diffusion_with_residual_head import (
        LatentDiffusionWithResidualHead,
    )

    mdl = LatentDiffusionWithResidualHead(
        backbone={
            "base_channels": 32,
            "timesteps": 10,
            "in_channels": 1,
            "out_channels": 1,
            "latent_channels": 4,
        },
        residual_head={"base_channels": 32, "depth": 3, "contrast_conditioning": True},
        num_contrasts=4,
    )
    ratio = mdl.residual_head_param_ratio()
    assert ratio < 0.10, (
        f"Residual head must be < 10 % of backbone parameters "
        f"(got {ratio:.2%}); the spec calls for a lightweight head."
    )


def test_residual_head_is_initialised_as_identity() -> None:
    """Zero-init of the head's output conv means the wrapper starts as backbone-only.

    This avoids destabilising a pretrained backbone at the moment the
    head is attached -- the head only contributes once it has trained.
    """
    from spectramr.models.generators.latent_diffusion_with_residual_head import (
        _ResidualHeadUNet,
    )

    head = _ResidualHeadUNet(in_channels=1, out_channels=1, base_channels=8, depth=2)
    x = torch.randn(1, 1, 16, 16)
    out = head(x)
    # Output conv weights are zero -> the residual must be all-zero at init.
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_residual_head_unet_forward_shape() -> None:
    """The head must preserve the input spatial shape."""
    from spectramr.models.generators.latent_diffusion_with_residual_head import (
        _ResidualHeadUNet,
    )

    head = _ResidualHeadUNet(in_channels=1, out_channels=1, base_channels=8, depth=2)
    x = torch.randn(2, 1, 16, 16)
    out = head(x)
    assert out.shape == x.shape


def test_residual_head_unet_accepts_contrast_embedding() -> None:
    """FiLM modulation must accept and consume a contrast embedding."""
    from spectramr.models.generators.latent_diffusion_with_residual_head import (
        _ResidualHeadUNet,
    )

    head = _ResidualHeadUNet(
        in_channels=1, out_channels=1, base_channels=8, depth=2, contrast_dim=4
    )
    x = torch.randn(2, 1, 16, 16)
    emb = torch.randn(2, 4)
    out = head(x, emb)
    assert out.shape == x.shape


class _FakeLatentBackbone(torch.nn.Module):
    """Latent geometry of LatentDiffusionGenerator (4 latent channels, factor 4), no weights."""

    config = None
    diffusion = object()

    def __init__(self) -> None:
        super().__init__()
        self.schedule: dict | None = None

    def forward(self, x, timesteps, context=None, condition_image=None):
        return torch.zeros_like(x)

    def decode_from_latent(self, z):
        return torch.nn.functional.interpolate(z[:, :1], scale_factor=4.0)

    def set_diffusion_schedule(self, **kwargs):
        self.schedule = kwargs

    def sample(self, shape, device="cpu", contrast_idx=None, condition_image=None,
               num_inference_steps=None):
        return torch.ones(shape)


def _wrapper(contrast: bool = True):
    from spectramr.models.generators.latent_diffusion_with_residual_head import (
        LatentDiffusionWithResidualHead,
        _ResidualHeadUNet,
    )

    m = LatentDiffusionWithResidualHead.__new__(LatentDiffusionWithResidualHead)
    torch.nn.Module.__init__(m)
    m.backbone = _FakeLatentBackbone()
    m.residual_head = _ResidualHeadUNet(
        in_channels=1, out_channels=1, base_channels=8, depth=2, contrast_dim=3 if contrast else 0
    )
    m.contrast_embedding = torch.nn.Embedding(3, 3) if contrast else None
    return m


def test_forward_keeps_the_latent_geometry() -> None:
    """#1066: forward maps z_t to a latent eps; the head never sees it."""
    z_t = torch.randn(2, 4, 8, 8)
    eps = _wrapper()(z_t, timesteps=torch.zeros(2, dtype=torch.long))
    assert eps.shape == z_t.shape


def test_decode_and_refine_scores_in_image_space() -> None:
    """#1066: the head receives the decoded x0 estimate and returns an image."""
    out = _wrapper().decode_and_refine(torch.randn(2, 4, 8, 8), torch.tensor([0, 2]))
    assert out.shape == (2, 1, 32, 32)


def test_refine_refuses_a_latent() -> None:
    with pytest.raises(ValueError, match="decoded image"):
        _wrapper().refine(torch.randn(2, 4, 8, 8), torch.tensor([0, 1]))


def test_conditioned_head_raises_without_contrast() -> None:
    with pytest.raises(ValueError, match="contrast_idx is None"):
        _wrapper().decode_and_refine(torch.randn(1, 4, 8, 8), None)


def test_sample_refines_the_backbone_sample() -> None:
    out = _wrapper(contrast=False).sample((1, 1, 16, 16))
    assert torch.equal(out, torch.ones(1, 1, 16, 16))  # zero-init head is the identity


def test_reverse_schedule_binds_through_the_wrapper() -> None:
    """#204 on this model: bind_latent_reverse_schedule must reach the backbone."""
    from spectramr.infrastructure.training.utils.diffusion_mixin import (
        bind_latent_reverse_schedule,
    )

    m = _wrapper()
    sched = type("S", (), {"num_timesteps": 10, "beta_schedule": "cosine", "device": "cpu",
                           "betas": torch.linspace(1e-4, 0.02, 10)})()
    assert bind_latent_reverse_schedule(m, sched)
    assert m.backbone.schedule["beta_schedule"] == "cosine"
