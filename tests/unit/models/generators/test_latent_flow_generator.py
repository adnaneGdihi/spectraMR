"""LatentFlowGenerator: exact coupling-flow density and the paired-VAE decode seam."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

import spectramr.models.generators.latent_flow_generator  # noqa: F401  (registers latent_flow)
from spectramr.infrastructure.training.builders.model_builder import _autoencoder_loader
from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory
from spectramr.models.generators.autoencoder_pretrain_generator import (
    AutoencoderPretrainGenerator,
)
from spectramr.models.generators.latent_flow_generator import FlowBlock, LatentFlowGenerator
from spectramr.models.registry import get_model_class

# 16x16 image -> AutoencoderKL (f=8) latent 4x2x2 = 16 values.
_PAIRED = {
    "latent_dim": 16,
    "flow_layers": 3,
    "hidden_dim": 16,
    "paired_vae": True,
    "latent_channels": 4,
    "base_channels": 32,
    "vae_checkpoint_path": "stage1/best.safetensors",
}


def _loaded(model: LatentFlowGenerator) -> LatentFlowGenerator:
    stage1 = AutoencoderPretrainGenerator(latent_channels=4, base_channels=32)
    _autoencoder_loader(model)(stage1.state_dict(), freeze=True, source="stage1")
    return model


def test_log_prob_matches_autograd_jacobian():
    torch.manual_seed(0)
    flow = LatentFlowGenerator(latent_dim=4, flow_layers=3, hidden_dim=8)
    x = torch.randn(1, 4)
    z, _ = flow.flow_inverse(x)
    jac = torch.autograd.functional.jacobian(lambda v: flow.flow_inverse(v)[0], x).reshape(4, 4)
    expected = -0.5 * (z**2).sum() - 2 * math.log(2 * math.pi) + torch.linalg.slogdet(jac).logabsdet
    assert torch.allclose(flow.log_prob(x)[0], expected, atol=1e-5)


def test_forward_inverse_round_trip_and_both_halves_move():
    torch.manual_seed(0)
    flow = LatentFlowGenerator(latent_dim=6, flow_layers=2, hidden_dim=8)
    z = torch.randn(3, 6)
    x, ld_f = flow.flow_forward(z)
    z_back, ld_i = flow.flow_inverse(x)
    assert torch.allclose(z_back, z, atol=1e-5)
    assert torch.allclose(ld_f, -ld_i, atol=1e-5)
    # The reversal between blocks means neither half passes through untouched.
    assert not torch.allclose(x[:, :3], z[:, :3]) and not torch.allclose(x[:, 3:], z[:, 3:])


def test_mismatched_vector_raises_instead_of_identity():
    flow = LatentFlowGenerator(latent_dim=16, flow_layers=1, hidden_dim=8)
    with pytest.raises(ValueError, match="paired_vae: true"):
        flow(torch.randn(2, 1, 8, 8))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"conditioning": "global"}, "conditioning"),
        ({"conditioning": "input_latent"}, "needs paired_vae"),
        ({"paired_vae": "yes"}, "true/false"),
    ],
)
def test_illegal_knobs_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        LatentFlowGenerator(latent_dim=16, **kwargs)


def test_odd_dim_raises():
    with pytest.raises(ValueError, match="even"):
        FlowBlock(5)


def test_unfrozen_vae_and_missing_checkpoint_are_refused():
    with pytest.raises(ValueError, match="freeze_vae"):
        LatentFlowGenerator(**_PAIRED, freeze_vae=False)
    with pytest.raises(ValueError, match="vae_checkpoint_path"):
        LatentFlowGenerator(**{**_PAIRED, "vae_checkpoint_path": None})


def test_stage1_load_freezes_and_decode_returns_image_shape():
    model = _loaded(LatentFlowGenerator(**_PAIRED))
    assert all(not p.requires_grad for p in model.autoencoder.parameters())
    assert model.decode_from_latent(torch.zeros(1, 16)).shape == (1, 1, 16, 16)


def test_option_schema_advertises_the_conditioning_set():
    assert LatentFlowGenerator.OPTION_SCHEMA["conditioning"] == ["none", "input_latent"]


def test_reachability_registry_strategy_and_decoded_validation_output():
    """registry -> builder VAE loader -> generative strategy NLL -> decoded image."""
    torch.manual_seed(0)
    model = _loaded(get_model_class("latent_flow")(conditioning="input_latent", **_PAIRED))
    cfg = SimpleNamespace(training=SimpleNamespace(strategy_class=None, training_mode="generative"))
    strategy_cls = TrainingStrategyFactory().get_strategy_class(cfg)
    strategy = strategy_cls.__new__(strategy_cls)

    target, aliased = torch.randn(2, 1, 16, 16), torch.randn(2, 1, 16, 16)
    seen = {}
    real_log_prob = model.log_prob

    def spy(x, condition=None):
        seen["condition"] = condition
        return real_log_prob(x, condition=condition)

    model.log_prob = spy
    out = strategy._compute_losses_impl(aliased, target, epoch=0, model=model)
    assert seen["condition"] is aliased
    assert torch.isfinite(out["loss_nll"])
    out["loss"].backward()
    assert all(p.grad is None for p in model.autoencoder.parameters())
    assert any(p.grad is not None for p in model.flow_blocks.parameters())
    assert model.cond_net[0].weight.grad is not None

    # Validation runs MetricsMixin.validation_step, which scores this forward's output.
    assert strategy_cls.validation_step.__qualname__ == "MetricsMixin.validation_step"
    strategy.env = SimpleNamespace(generator=model.eval())
    with torch.no_grad():
        pred = strategy._validation_generator_forward(aliased)
    assert pred.shape == target.shape
