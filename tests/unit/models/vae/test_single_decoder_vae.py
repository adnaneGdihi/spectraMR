"""``single_decoder_vae``: the width-matched control for ``moe_vae``."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.vae.moe_vae import MoEVAE
from spectramr.models.vae.single_decoder_vae import SingleDecoderVAE

CONTROL = "experiments/inprogress/vae_latent/moe_vae_single_decoder_matched.yaml"
TREATMENT = "experiments/inprogress/vae_latent/moe_vae.yaml"


def _count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _build_arm(path: str) -> torch.nn.Module:
    """The production build: settings -> kwarg resolution -> GeneratorBuilder -> registry."""
    from spectramr.config.settings import TrainingSettings
    from spectramr.infrastructure.builders.generator_kwargs import resolve_generator_kwargs
    from spectramr.infrastructure.builders.leaf.model_builders import GeneratorBuilder

    cfg = TrainingSettings.from_yaml(path)
    builder = (
        GeneratorBuilder(cfg, "cpu")
        .with_architecture(cfg.model.model_type)
        .with_input_channels(cfg.model.in_channels)
        .with_output_channels(cfg.model.out_channels)
    )
    for key, value in resolve_generator_kwargs(cfg, device="cpu").items():
        builder.with_parameter(key, value)
    return builder.validate().build()


def test_arm_builds_the_control_with_the_declared_decoder_width():
    model = _build_arm(CONTROL)
    assert isinstance(model, SingleDecoderVAE)
    assert (model.hidden, model.decoder_hidden) == (128, 288)


def test_control_matches_the_moe_parameter_count_stated_in_the_arm():
    control, moe = _build_arm(CONTROL), _build_arm(TREATMENT)
    assert isinstance(moe, MoEVAE)
    assert _count(control.decoder) == 1_297_009
    assert _count(moe.experts) == 1_322_756
    assert (_count(control), _count(moe)) == (1_691_953, 1_717_960)
    # Same trunk and latent heads: only the decoder side differs.
    for name in ("encoder", "fc_mu", "fc_logvar"):
        assert _count(getattr(control, name)) == _count(getattr(moe, name))
    assert abs(_count(control) / _count(moe) - 1.0) < 0.02


def test_forward_returns_the_moe_contract_without_aux_loss():
    model = SingleDecoderVAE(latent_dim=8, hidden=32, decoder_hidden=48)
    x = torch.randn(3, 1, 32, 32)
    out = model(x)
    assert set(out) == {"reconstruction", "kl_override"}
    assert out["reconstruction"].shape == x.shape
    assert float(out["kl_override"].detach()) >= 0.0
    (out["reconstruction"].pow(2).mean() + out["kl_override"]).backward()
    assert model.decoder.fc.weight.grad is not None


def test_decoder_width_defaults_to_the_trunk_width():
    assert SingleDecoderVAE(hidden=32).decoder_hidden == 32


@pytest.mark.parametrize("kwargs", [{"decoder_hidden": 40}, {"hidden": 24}, {"decoder_hidden": 0}])
def test_width_not_a_multiple_of_16_raises(kwargs):
    with pytest.raises(ValueError, match="multiple of 16"):
        SingleDecoderVAE(**kwargs)


def test_unknown_kwarg_raises():
    with pytest.raises(TypeError):
        SingleDecoderVAE(num_experts=1)  # type: ignore[call-arg]
