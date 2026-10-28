"""Regression guard for the dimension-contract back-fill.

The high-traffic models below were given ModelCapabilities declarations
(2026-06-24, Layer-1 of the dimension-contract plan) so the audit can check
their dimension flow instead of fail-soft skipping. This test pins those
declarations so a future refactor can't silently drop them back to ``None``
(which would re-open the default-deny gap).
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def caps_for():
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import get_model_capabilities

    populate_model_registry()
    return get_model_capabilities


# Every back-filled model must declare 2D spatial_dims (the enforcement-critical
# field that lifts it out of the default-deny bucket).
#
# Batch 1 (2026-06-24): the ten highest-traffic models.
# Batch 2 (2026-06-24): the next verified-2D tier (each confirmed [B, C, H, W]
# via its forward / Conv2d usage before declaring — a wrong contract is worse
# than none). The 0-conv wrapper models (cold_diffusion / kan_gan /
# deep_image_prior) were intentionally deferred: their rank depends on the
# inner backbone, so they are NOT declared here.
_SPATIAL_2D = [
    # batch 1
    "complex_unet",
    "unet",
    # Was `enhanced_unet` until #993 de-collided that name: the SE+residual
    # variant this entry has always described is now registered as
    # `se_residual_unet`, while `enhanced_unet` became an alias of
    # `configurable_unet` (spatial_dims=(2, 3), image->image) to match what the
    # factory had been building for it all along. Following the MODEL, not the
    # string, is what keeps this entry meaning what it was written to mean.
    "se_residual_unet",
    "nafnet",
    "vae",
    "mamba",
    "variational_network",
    "diff_varnet",
    "mamba_unet",
    "enhanced_kan_unet",
    # batch 2
    "kspace_gpt",
    "diffusion_unet",
    "swin_diff_rec",
    "varnet",
    "attention_unet",
    "vit_kan",
    "refined_kan_unet",
    "diff_varnet_kan",
    "multicontrast_fusion",
    "d2_mamba",
]


# Batch-2 models intentionally NOT yet declared (wrapper models whose spatial
# rank is set by an inner backbone) — pinned so a future contributor does not
# mistakenly declare a fixed rank without first resolving the wrapper.
_DEFERRED_WRAPPERS = ["cold_diffusion", "kan_gan", "deep_image_prior"]


@pytest.mark.parametrize("name", _SPATIAL_2D)
def test_backfilled_model_declares_2d(name, caps_for):
    caps = caps_for(name)
    assert caps is not None, f"{name} lost its capability declaration"
    assert caps.spatial_dims == (2,), f"{name} spatial_dims={caps.spatial_dims}"


def test_complex_kspace_models_declare_complex_contract(caps_for):
    """complex_unet / diff_varnet are interleaved-complex k-space nets."""
    for name in ("complex_unet", "diff_varnet"):
        caps = caps_for(name)
        assert caps.input_domain == "kspace"
        assert caps.output_domain == "kspace"
        assert caps.accepts_complex is True
        assert caps.expects_real_imag_interleaved is True


def test_vae_declares_the_decoded_image_as_its_output(caps_for):
    # forward returns (reconstruction, mu, log_var); the loss sees the decoded image.
    caps = caps_for("vae")
    assert caps.output_domain == "image"


def test_agnostic_models_do_not_declare_a_domain(caps_for):
    """Domain-agnostic denoisers declare spatial_dims but NOT a domain.

    Declaring a domain on a generic UNet would make the loss/metric domain-chain
    checks false-fire — so they intentionally leave domain None.
    """
    # `se_residual_unet` was spelled `enhanced_unet` until #993 — see the note in
    # `_SPATIAL_2D`. `enhanced_unet` is deliberately NOT in this list any more:
    # it now aliases `configurable_unet`, which DOES declare image->image, so
    # asserting `input_domain is None` over it would assert the opposite of what
    # that model advertises.
    for name in ("unet", "se_residual_unet", "nafnet", "mamba", "enhanced_kan_unet"):
        caps = caps_for(name)
        assert caps.input_domain is None
        assert caps.output_domain is None


@pytest.mark.parametrize("name", _DEFERRED_WRAPPERS)
def test_deferred_wrapper_models_stay_undeclared(name, caps_for):
    """Wrapper models (cold_diffusion / kan_gan / deep_image_prior) must NOT
    declare a fixed ``spatial_dims`` until the inner backbone's rank is resolved.

    Declaring (2,) blindly would assert a contract the wrapper can't honour for
    a 3D backbone. This pins the deferral so a back-fill sweep doesn't guess.
    """
    caps = caps_for(name)
    declared_rank = caps is not None and caps.spatial_dims is not None
    assert not declared_rank, (
        f"{name} is a wrapper model whose rank depends on its inner backbone; "
        "it must stay undeclared until that is resolved (see batch-2 notes)."
    )
