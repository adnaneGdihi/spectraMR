"""KANVAEGenerator exposes its posterior so the VAE strategy's KL is not zero (#2362)."""

from __future__ import annotations

import torch

from spectramr.models.generators.kan_vae_generator import KANVAEGenerator


def test_forward_caches_the_posterior() -> None:
    m = KANVAEGenerator(in_channels=1, out_channels=1, latent_dim=8, kan_hidden_dim=8).train()
    assert m.last_aux() is None
    x = torch.randn(2, 1, 32, 32)
    out = m(x)
    assert out.shape == x.shape
    aux = m.last_aux()
    assert aux["mu"].shape == (2, 8) and aux["logvar"].shape == (2, 8)
    assert aux["mu"].requires_grad
