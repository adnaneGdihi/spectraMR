"""NoiseAdaptiveScoreStrategy: the DSM term fires, sigma reaches the denoiser, the sampler is the reverse SDE.

The strategy is resolved through ``TrainingStrategyFactory.STRATEGY_CLASS_PATHS`` (the
production path for ``training_mode: noise_adaptive_score``) and driven on a real
``ColdDiffusionGenerator`` wrapping a tiny time-conditioned denoiser. Construction uses
``__new__`` plus attribute injection, the idiom of the other promoted-strategy tests.
"""

from __future__ import annotations

import importlib
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory
from spectramr.models.generators.cold_diffusion_generator import ColdDiffusionGenerator

C, H, W = 2, 8, 8


class _TinyDenoiser(nn.Module):
    """forward(x, t): one conv plus a t-dependent bias; records every t it is fed."""

    consumes_time = True

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2 * C, C, 3, padding=1)
        self.t_scale = nn.Parameter(torch.ones(1))
        self.seen_t: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.seen_t.append(t.detach().clone())
        return self.conv(x) + self.t_scale * t.view(-1, 1, 1, 1)


class _TimeBlind(nn.Module):
    consumes_time = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :C]


def _strategy_class() -> type:
    path = TrainingStrategyFactory.STRATEGY_CLASS_PATHS["noise_adaptive_score"]
    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


def _make(denoiser: nn.Module) -> tuple[object, ColdDiffusionGenerator]:
    gen = ColdDiffusionGenerator(denoising_model=denoiser, timesteps=10)
    cls = _strategy_class()
    s = cls.__new__(cls)
    s.env = SimpleNamespace(generator=gen)
    s.config = SimpleNamespace(
        model=SimpleNamespace(out_channels=C),
        validation=SimpleNamespace(sampling=SimpleNamespace(steps=3)),
        run=SimpleNamespace(seed=0),
    )
    s._prepare_batch_context = lambda inp, tgt, **kw: {"hr": tgt}
    s._prepare_generator_inputs = lambda ctx, inp, validation=False: (inp, {})
    s.declared = []
    s._declare_model_input = lambda tensors, **kw: s.declared.append((tensors, kw))
    return s, gen


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return torch.randn(2, C, H, W), torch.randn(2, C, H, W)


def test_dsm_fires_and_trains_the_denoiser_through_sigma() -> None:
    denoiser = _TinyDenoiser()
    s, _ = _make(denoiser)
    y, x0 = _batch()
    out = s._compute_losses_impl(input_batch=y, target_batch=x0, epoch=0)

    assert "loss_score_dsm" in out and torch.isfinite(out["g_total_loss"])
    # One call, fed the float c_noise = ln(sigma)/4, not an integer cold-diffusion timestep.
    (t,) = denoiser.seen_t
    assert t.is_floating_point() and t.shape == (2,)
    sigma = torch.exp(4 * t)
    assert ((sigma >= s.sigma_min * 0.999) & (sigma <= s.sigma_max * 1.001)).all()
    # The network input is [c_in x_sigma, y]: the measurement rides as the second half.
    model_input = s.declared[0][0]["model_input"]
    assert model_input.shape[1] == 2 * C and torch.equal(model_input[:, C:], y)

    out["g_total_loss"].backward()
    assert denoiser.conv.weight.grad is not None and denoiser.conv.weight.grad.abs().sum() > 0
    assert denoiser.t_scale.grad is not None and denoiser.t_scale.grad.abs() > 0


def test_snr_map_reweights_the_dsm_term() -> None:
    s, _ = _make(_TinyDenoiser())
    y, x0 = _batch()
    torch.manual_seed(1)
    plain = s._compute_losses_impl(input_batch=y, target_batch=x0)["loss_score_dsm"]
    torch.manual_seed(1)
    weighted = s._compute_losses_impl(
        input_batch=y, target_batch=x0, batch={"snr_map": torch.full((2, C, H, W), 2.0)}
    )["loss_score_dsm"]
    assert torch.allclose(weighted, 4.0 * plain)


def test_time_blind_denoiser_is_refused() -> None:
    s, _ = _make(_TimeBlind())
    y, x0 = _batch()
    with pytest.raises(ValueError, match="time-conditioned denoiser"):
        s._compute_losses_impl(input_batch=y, target_batch=x0)


def test_channel_mismatch_is_refused() -> None:
    s, _ = _make(_TinyDenoiser())
    y, x0 = _batch()
    with pytest.raises(ValueError, match="out_channels"):
        s._compute_losses_impl(input_batch=y, target_batch=x0[:, :1])


def test_validation_samples_the_reverse_sde_for_the_declared_steps() -> None:
    denoiser = _TinyDenoiser()
    s, _ = _make(denoiser)
    y, _ = _batch()
    with torch.no_grad():
        out = s._validation_forward(y, {})
    assert out.shape == (2, C, H, W) and torch.isfinite(out).all()
    assert len(denoiser.seen_t) == 3


def test_reverse_sde_contracts_onto_a_point_mass() -> None:
    # Data = δ(0): the exact score is -x/sigma², so the reverse SDE must end near 0.
    s, _ = _make(_TinyDenoiser())
    g = torch.Generator().manual_seed(0)
    x = s.reverse_sde_sample(
        lambda x, sig: -x / sig.view(-1, 1, 1, 1) ** 2, (4, 1, 8, 8), n_steps=200, generator=g
    )
    assert x.abs().max() < 10 * s.sigma_min * math.sqrt(200)


def test_standalone_infer_is_refused() -> None:
    assert _strategy_class().standalone_infer_blocker(None)
