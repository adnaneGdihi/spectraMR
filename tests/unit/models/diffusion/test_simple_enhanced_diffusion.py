"""Unit tests for SimpleEnhancedDiffusionUNet knob validation.

Pins the pitfall-#9 guards on ``time_embedding_type`` and ``noise_type``: an
unknown value must raise at construction instead of silently resolving to the
sinusoidal/gaussian defaults (a YAML advertising ``noise_type: rician`` would
otherwise run a plain gaussian model without any signal that the knob was
dropped).
"""

import pytest
import torch

from spectramr.models.diffusion.simple_enhanced_diffusion import (
    SimpleChiSquareDiffusionUNet,
    SimpleEnhancedDiffusionUNet,
    SimpleGaussianDiffusionUNet,
    SimpleGaussianKANDiffusionUNet,
    SimpleKANTimeEmbedding,
    SimpleTimeEmbedding,
)


def _tiny_model(**kwargs) -> SimpleEnhancedDiffusionUNet:
    kwargs.setdefault("model_channels", 8)
    kwargs.setdefault("channel_mult", (1,))
    kwargs.setdefault("num_res_blocks", 1)
    return SimpleEnhancedDiffusionUNet(in_channels=1, out_channels=1, **kwargs)


@pytest.mark.parametrize("noise_type", ["gaussian", "chi_square", "gaussian_kan"])
def test_valid_noise_types_construct_and_run(noise_type):
    model = _tiny_model(noise_type=noise_type)
    x = torch.randn(1, 1, 8, 8)
    t = torch.tensor([5])
    out = model(x, t)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "time_embedding_type,expected",
    [("sinusoidal", SimpleTimeEmbedding), ("kan", SimpleKANTimeEmbedding)],
)
def test_valid_time_embedding_types_construct(time_embedding_type, expected):
    model = _tiny_model(time_embedding_type=time_embedding_type)
    assert isinstance(model.time_embed, expected)


def test_unknown_noise_type_raises():
    with pytest.raises(
        ValueError, match="'gaussian', 'chi_square', 'gaussian_kan'"
    ):
        _tiny_model(noise_type="rician")


def test_unknown_time_embedding_type_raises():
    with pytest.raises(ValueError, match="'sinusoidal', 'kan'"):
        _tiny_model(time_embedding_type="fourier")


@pytest.mark.parametrize(
    "subclass",
    [
        SimpleGaussianDiffusionUNet,
        SimpleChiSquareDiffusionUNet,
        SimpleGaussianKANDiffusionUNet,
    ],
)
def test_registered_variants_still_construct(subclass):
    model = subclass(
        in_channels=1,
        out_channels=1,
        model_channels=8,
        channel_mult=(1,),
        num_res_blocks=1,
    )
    assert model.noise_type in ("gaussian", "chi_square", "gaussian_kan")


@pytest.mark.parametrize("horizon", [29, 1000])
def test_the_kan_time_code_divides_by_its_horizon(horizon):
    """PLANTED (#2442): the KAN code divided by 1000 whatever the chain length."""
    embed = SimpleKANTimeEmbedding(16, horizon)
    seen = []
    embed.kan_layers.register_forward_pre_hook(lambda _m, args: seen.append(args[0]))
    embed(torch.tensor([0, horizon - 1]))
    assert torch.equal(seen[0].squeeze(-1), torch.tensor([0, horizon - 1]).float() / horizon)


def test_the_kan_variant_names_its_horizon_for_the_builder():
    """The builder injects only into constructors that name num_timesteps."""
    from spectramr.infrastructure.builders.generator_kwargs import resolve_contract

    contract = resolve_contract(model_cls=SimpleGaussianKANDiffusionUNet, model_type=None)
    assert "num_timesteps" in contract.accepted
    model = SimpleGaussianKANDiffusionUNet(
        model_channels=8, channel_mult=(1,), num_res_blocks=1, num_timesteps=29
    )
    assert model.time_embed.num_timesteps == 29
