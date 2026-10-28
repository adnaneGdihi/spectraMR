"""Regression tests for ``ConfigurableVAE``.

Two contracts are locked here:

1. The VAE forward/compute_loss path hard-assumes a Gaussian latent (it reads
   ``mu``/``logvar``). A ConfigurableVAE built with a non-gaussian
   ``LatentConfig`` previously raised a bare ``KeyError`` deep in ``forward``;
   it must now fail fast at construction with a clear ``NotImplementedError``
   naming the offending distribution (design-smell -> explicit guard).

2. The pre-refactor ``src.domain.entities.models.*`` import paths in
   ``USAGE.md`` no longer resolve after the 2026-05 ``src -> spectramr`` refactor.
   The doc must use ``spectramr.<sub>`` paths, and the documented symbols must
   import cleanly so the doc cannot drift from the package layout.

CPU-only, tiny tensors, fixed seed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from spectramr.models.latent.configurable_vae import (
    ConfigurableVAE,
    create_configurable_vae,
)

USAGE = Path(__file__).resolve().parents[4] / "src/spectramr/models/latent/USAGE.md"


@pytest.mark.parametrize("distribution", ["vq", "categorical", "flow"])
def test_non_gaussian_distribution_raises_at_construction(distribution: str) -> None:
    from spectramr.models.latent.configurable_architecture import LatentConfig

    with pytest.raises(NotImplementedError) as excinfo:
        ConfigurableVAE(latent_config=LatentConfig(distribution=distribution))
    msg = str(excinfo.value)
    assert "gaussian" in msg
    assert distribution in msg


def test_simple_preset_forward_round_trip() -> None:
    torch.manual_seed(0)
    vae = ConfigurableVAE.from_presets("simple")  # gaussian by default
    x = torch.randn(2, 1, 256, 256)
    recon, mu, logvar, z = vae.forward(x, return_latent=True)
    assert recon.shape == x.shape
    assert mu.shape == logvar.shape == (2, vae.latent_config.latent_dim)


def test_create_configurable_vae_gaussian_constructs() -> None:
    vae = create_configurable_vae(preset="simple", latent_dim=128)
    assert vae.latent_config.distribution == "gaussian"
    assert vae.latent_config.latent_dim == 128


# --- USAGE.md doc-contract regression -------------------------------------


def test_usage_md_has_no_pre_refactor_imports() -> None:
    text = USAGE.read_text()
    assert "from src." not in text
    assert "src.domain.entities" not in text
    assert "from spectramr.models.latent" in text


def test_usage_md_documented_symbols_import_cleanly() -> None:
    # Guards against future drift between the doc and the package layout.
    from spectramr.models.latent import (  # noqa: F401
        DecoderConfig,
        EncoderConfig,
        LatentConfig,
    )
    from spectramr.models.latent.configurable_vae import (  # noqa: F401
        ConfigurableVAE as _CV,
    )
    from spectramr.models.latent.configurable_vae import (  # noqa: F401
        create_configurable_vae as _ccv,
    )


# --- #801: the VAE is an nn.Module and its encoder/decoder register ----------


def test_the_vae_is_an_nn_module() -> None:
    """It implemented ``IGenerator`` only, which carries no ``nn.Module``."""
    from torch import nn

    assert issubclass(ConfigurableVAE, nn.Module)
    assert ConfigurableVAE.__call__ is nn.Module.__call__


def test_encoder_and_decoder_register_so_the_vae_can_be_trained() -> None:
    """Both halves were plain attributes, so ``parameters()`` was empty."""
    vae = ConfigurableVAE.from_presets("simple")
    children = dict(vae.named_children())
    assert "encoder" in children
    assert "decoder" in children
    assert sum(1 for _ in vae.parameters()) > 0
    keys = vae.state_dict()
    assert any(k.startswith("encoder.") for k in keys)
    assert any(k.startswith("decoder.") for k in keys)
