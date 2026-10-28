"""XDiffusionGenerator embeds the strategy's step on the arm's horizon (#2440).

It took no timestep: training drew ``randint(0, 1000)`` per forward and eval used
0, so the time code carried nothing about the corruption level of ``x_t``, and
the 3D path fed its features through the time MLP instead of a time code.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch

from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.builders.generator_kwargs import resolve_generator_kwargs
from spectramr.models.generators import x_diffusion_generator as module
from spectramr.models.generators.x_diffusion_generator import XDiffusionGenerator

ARM = "experiments/inprogress/cross_modal_diffusion/exp_x_diffusion_cross_modal.yaml"


def _generator(num_timesteps: int | None = 100) -> XDiffusionGenerator:
    torch.manual_seed(0)
    return XDiffusionGenerator(
        resolution_2d=(32, 32),
        resolution_3d=(8, 16, 16),
        cross_attention_dim=32,
        conditioning_dim=16,
        num_layers=1,
        num_heads=2,
        num_timesteps=num_timesteps,
    )


@pytest.fixture
def sinusoid_calls(monkeypatch):
    calls = []
    real = module.sinusoidal_timestep_embedding

    def spy(timesteps, dim, **kwargs):
        calls.append((timesteps.tolist(), kwargs.get("max_timesteps"), kwargs.get("scale")))
        return real(timesteps, dim, **kwargs)

    monkeypatch.setattr(module, "sinusoidal_timestep_embedding", spy)
    return calls


@pytest.mark.parametrize("shape", [(2, 1, 32, 32), (2, 1, 8, 16, 16)])
def test_the_strategys_step_reaches_the_horizon_band_sinusoid(shape, sinusoid_calls):
    """PLANTED: both paths embedded something other than the step they were given."""
    _generator()(torch.randn(*shape), timesteps=torch.tensor([3, 97]))
    assert sinusoid_calls == [([3, 97], 100.0, "horizon_band")]


@pytest.mark.parametrize("shape", [(2, 1, 32, 32), (2, 1, 8, 16, 16)])
def test_the_output_depends_on_the_step(shape):
    gen, x = _generator(), torch.randn(*shape)
    early = gen(x, timesteps=torch.tensor([3, 3]))
    late = gen(x, timesteps=torch.tensor([90, 90]))
    assert not torch.equal(early, late)


def test_a_training_forward_without_a_step_is_deterministic():
    """PLANTED: training mode drew a fresh randint(0, 1000) on every call."""
    gen, x = _generator().train(), torch.randn(2, 1, 32, 32)
    assert torch.equal(gen(x), gen(x))


def test_no_step_is_the_clean_endpoint():
    gen, x = _generator(), torch.randn(2, 1, 32, 32)
    assert torch.equal(gen(x), gen(x, timesteps=torch.zeros(2, dtype=torch.long)))


def test_a_step_without_a_horizon_raises_rather_than_assuming_one():
    with pytest.raises(ValueError, match="needs num_timesteps"):
        _generator(None)(torch.randn(2, 1, 32, 32), timesteps=torch.tensor([3, 3]))


def test_a_step_count_that_is_not_the_batch_raises():
    with pytest.raises(ValueError, match="3 timesteps for a batch of 2"):
        _generator()(torch.randn(2, 1, 32, 32), timesteps=torch.tensor([1, 2, 3]))


def test_the_arm_resolves_its_declared_horizon():
    """The arm declares T=100 in model_kwargs.timesteps and training.diffusion.timesteps."""
    cfg = TrainingSettings.from_yaml(ARM)
    kwargs = resolve_generator_kwargs(
        config=cfg, model_cls=XDiffusionGenerator, model_type="x_diffusion", device="cpu"
    )
    assert kwargs["num_timesteps"] == cfg.training.diffusion.timesteps == 100
    assert "timesteps" not in kwargs
