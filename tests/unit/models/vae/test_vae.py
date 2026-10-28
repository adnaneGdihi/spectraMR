"""``VAE`` (``models/vae/vae.py``): what its forward returns and how it is registered."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.init_registry import populate_model_registry
from spectramr.models.registry import get_model_capabilities
from spectramr.models.vae.vae import VAE


def test_forward_returns_the_decoded_image() -> None:
    torch.manual_seed(0)
    model = VAE(in_channels=1, latent_dim=8, hidden_dims=[8, 16, 16, 16]).eval()
    x = torch.rand(2, 1, 32, 32)  # 2 * 2**4
    with torch.no_grad():
        reconstruction, mu, log_var = model(x)
    assert reconstruction.shape == x.shape
    assert mu.shape == log_var.shape == (2, 8)


def test_registration_declares_an_image_output() -> None:
    """The tensor a loss sees is the decoded image, so the declared output
    domain is ``image``; ``latent`` (until 2026-09-03) failed every arm that
    declared the image-domain reconstruction loss a VAE is trained with."""
    populate_model_registry()
    caps = get_model_capabilities("vae")
    assert (
        caps is not None
        and str(getattr(caps.output_domain, "value", caps.output_domain)) == "image"
    )


def test_a_patch_the_decoder_cannot_render_is_refused() -> None:
    """Planted violation: two hidden dims render 8x8, whatever the input size."""
    model = VAE(in_channels=1, latent_dim=8, hidden_dims=[8, 16]).eval()
    with (
        pytest.raises(ValueError, match="2 \\* 2\\*\\*len\\(hidden_dims\\) = 8"),
        torch.no_grad(),
    ):
        model(torch.rand(2, 1, 32, 32))


def _small(**kw) -> VAE:
    torch.manual_seed(0)
    return VAE(in_channels=1, out_channels=1, latent_dim=8, hidden_dims=(8, 8, 8), **kw).eval()


def test_the_deterministic_comparator_decodes_the_mean() -> None:
    model = _small(deterministic=True)
    x = torch.rand(2, 1, 16, 16)
    with torch.no_grad():
        recon, mu, _ = model(x)
        again, _, _ = model(x)
        torch.testing.assert_close(recon, again)
        torch.testing.assert_close(recon, model.decode(mu))


def test_the_variational_default_samples() -> None:
    model = _small()
    x = torch.rand(2, 1, 16, 16)
    with torch.no_grad():
        assert not torch.allclose(model(x)[0], model(x)[0])


def test_a_non_bool_deterministic_raises() -> None:
    with pytest.raises(TypeError, match="deterministic must be a bool"):
        _small(deterministic="yes")
