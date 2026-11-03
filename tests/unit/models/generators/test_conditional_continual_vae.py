"""ConditionalContinualVAEGenerator builds the declared depth and exposes its posterior."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.conditional_continual_vae import (
    ConditionalContinualVAEGenerator,
)


def _tiny(**kw) -> ConditionalContinualVAEGenerator:
    return ConditionalContinualVAEGenerator(in_channels=1, out_channels=1, num_tasks=1, **kw)


def test_declared_depth_and_widths_are_built() -> None:
    """#2371: features/encoder_blocks set the latent width and resolution."""
    m = _tiny(features=[4, 8, 16, 32, 64], encoder_blocks=4, decoder_blocks=4).eval()
    x = torch.randn(1, 1, 32, 32)
    out = m(x)
    assert out.shape == x.shape
    assert m.last_aux()["mu"].shape == (1, 64, 2, 2)

    shallow = _tiny(features=[4, 8], encoder_blocks=1, decoder_blocks=1).eval()
    shallow(x)
    assert shallow.last_aux()["mu"].shape == (1, 8, 16, 16)


@pytest.mark.parametrize(
    "kw",
    [
        {"features": [4, 8, 16], "encoder_blocks": 4, "decoder_blocks": 4},
        {"features": [4, 8], "encoder_blocks": 1, "decoder_blocks": 9},
    ],
)
def test_inconsistent_depth_raises(kw: dict) -> None:
    with pytest.raises(ValueError, match="conditional_continual_vae"):
        _tiny(**kw)


def test_posterior_is_cached_with_gradients() -> None:
    """#2362: the strategy's KL reads last_aux(); the cached posterior stays on the graph."""
    m = _tiny(features=[4, 8, 16], encoder_blocks=2, decoder_blocks=2).train()
    assert m.last_aux() is None
    m(torch.randn(2, 1, 16, 16))
    aux = m.last_aux()
    assert aux["mu"].requires_grad and aux["logvar"].requires_grad
    kl = -0.5 * torch.sum(1 + aux["logvar"] - aux["mu"].pow(2) - aux["logvar"].exp())
    kl.backward()
    assert m.fc_mu.weight.grad is not None
