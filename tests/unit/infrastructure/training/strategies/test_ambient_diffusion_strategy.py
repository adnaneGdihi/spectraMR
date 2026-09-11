r"""Tests for the Ambient Diffusion reconstruction strategy (Phase B step 5).

Ambient Diffusion (Daras et al. 2023; Aali et al. MRI A-DPS) learns a clean prior
from undersampled k-space alone by lifting the SSDU Λ/Θ split onto a diffusion
model. The pure core ``ambient_diffusion_step`` runs the diffusion forward
(noise → eps → x₀) on the Λ zero-filled surrogate; the held-out Θ consistency
(``ambient_consistency_residual``) supervises x₀ against the measurement.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.infrastructure.training.strategies.ambient_diffusion_strategy import (
    AmbientDiffusionStrategy,
    ambient_diffusion_step,
)
from spectramr.infrastructure.training.strategies.diffusion import (
    DiffusionTrainingStrategy,
)
from spectramr.models.generators.score_based_diffusion_generator import (
    ScoreBasedDiffusionGenerator,
)
from spectramr.models.losses.ambient_consistency_loss import ambient_consistency_residual


def _tiny_score_gen():
    torch.manual_seed(0)
    return ScoreBasedDiffusionGenerator(
        denoising_model=nn.Conv2d(1, 1, 3, padding=1), timesteps=10, in_channels=1
    )


def test_ambient_step_shapes_and_grad():
    gen = _tiny_score_gen()
    x_start = torch.randn(1, 1, 16, 16)
    t = torch.randint(0, 10, (1,))
    noise = torch.randn_like(x_start)
    eps_pred, x0_pred, x_t = ambient_diffusion_step(
        gen._model_forward, gen._predict_start_from_noise, gen.q_sample, x_start, t, noise
    )
    assert eps_pred.shape == x0_pred.shape == x_start.shape
    assert x0_pred.requires_grad

    # x_t is returned, not discarded, because it is the tensor the network is
    # fed and the strategy owes it to the debug-snapshot contract
    # (non-negotiable 14). Recomputing q_sample at the call site to recover it
    # would duplicate work every training step.
    assert torch.equal(x_t, gen.q_sample(x_start, t, noise)), (
        "the returned x_t must be the tensor the model actually received"
    )


def test_ambient_objective_trains_the_network():
    """Mechanism fires: denoise + held-out Θ consistency push gradient into the net."""
    gen = _tiny_score_gen()
    x_start = torch.randn(1, 1, 16, 16)
    t = torch.randint(0, 10, (1,))
    noise = torch.randn_like(x_start)
    eps_pred, x0_pred, _x_t = ambient_diffusion_step(
        gen._model_forward, gen._predict_start_from_noise, gen.q_sample, x_start, t, noise
    )
    y = fft2c(torch.complex(x_start, torch.zeros_like(x_start)))
    theta = torch.zeros(1, 1, 16, 16)
    theta[..., ::2] = 1.0
    loss = nn.functional.mse_loss(eps_pred, noise) + ambient_consistency_residual(x0_pred, y, theta)
    assert loss.requires_grad and float(loss.detach()) > 0
    gen.zero_grad(set_to_none=True)
    loss.backward()
    grad_sq = sum(float(p.grad.pow(2).sum()) for p in gen.parameters() if p.grad is not None)
    assert grad_sq > 0.0


def test_strategy_extends_diffusion():
    assert issubclass(AmbientDiffusionStrategy, DiffusionTrainingStrategy)


def test_strategy_wires_ambient_mechanism():
    import inspect

    src = inspect.getsource(AmbientDiffusionStrategy)
    assert "split_acquired_mask" in src  # SSDU Λ/Θ split lifted onto diffusion
    assert "ambient_consistency" in src.lower()
    assert "theta" in src.lower()


def test_it_declares_its_own_loss_ownership() -> None:
    """Issue #1918: its objective regresses the NOISE (mse_loss(eps_pred, noise)), not the target.

    Read off ``__dict__``, never the inherited value: this class sits under
    ``DiffusionTrainingStrategy``, whose ``folds_image_losses = True`` is truthful for ITSELF and
    becomes a lie the moment a subclass replaces ``_compute_losses_impl``. An
    inherited True makes the audit's ``image_losses_reach_the_objective`` witness
    PASS every declared ``losses.image_losses`` entry on this strategy's arms
    while the training step discards them.
    """
    assert AmbientDiffusionStrategy.__dict__["folds_image_losses"] is False
    assert AmbientDiffusionStrategy.__dict__["inline_losses"] == frozenset()
