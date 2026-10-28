"""Unit + regression tests for TeichmullerColdDiffusionStrategy.

Regression coverage for the four §6 findings:

* F1/F2: the strategy now applies a μ_t-parameterised SFC degrade→restore
  objective, so the schedule head receives a NON-ZERO, BATCH-DEPENDENT
  gradient (the old ``0.0 * μ`` no-op gave it none).
* F3: a missing ``training.teichmuller`` block RAISES at construction
  (no silent getattr-default fallback).
* F4: μ_T is a spatially-VARYING Beltrami field with ``|μ_T| < 1``
  everywhere (the old uniform-real ``0.5`` endpoint was degenerate).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from spectramr.config.schemas.training.teichmuller import (
    TeichmullerTrainingConfigSchema,
)
from spectramr.infrastructure.training.strategies.teichmuller_cold_diffusion_strategy import (
    TeichmullerColdDiffusionStrategy,
)
from tests.utils.mock_environment import create_mock_training_env


def _simple_model() -> torch.nn.Module:
    """Shape-preserving 1->1 channel conv generator."""
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(8, 1, kernel_size=3, padding=1),
    )


def _mock_config(teichmuller: TeichmullerTrainingConfigSchema | None) -> MagicMock:
    config = MagicMock()
    config.device = "cpu"
    # No parallel block: a bare MagicMock would read as an unknown parallel.strategy.
    config.parallel = None
    config.optimization = MagicMock()
    config.optimization.optimizer.learning_rate = 1e-3
    # schema defaults; bare MagicMock fails StandardOptimizerStepper's raise-on-unknown
    config.optimization.gradient.clip.enabled = False
    # mixed_precision.py validates amp_dtype; a bare MagicMock fails it
    config.optimization.precision.dtype = "float32"
    config.optimization.gradient.clip.method = "norm"
    config.optimization.gradient.clip.value = 1.0
    config.physics = None
    config.training = MagicMock()
    config.training.strategy_class = (
        "spectramr.infrastructure.training.strategies."
        "teichmuller_cold_diffusion_strategy.TeichmullerColdDiffusionStrategy"
    )
    # Explicit: a MagicMock would auto-create a child attr (never None).
    config.training.teichmuller = teichmuller
    return config


def _build_strategy(
    model: torch.nn.Module,
    teichmuller: TeichmullerTrainingConfigSchema | None,
) -> TeichmullerColdDiffusionStrategy:
    config = _mock_config(teichmuller)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    env = create_mock_training_env(
        config=config, device="cpu", generator=model, opt_g=optimizer
    )
    return TeichmullerColdDiffusionStrategy(env)


def _schedule_grad_vector(strategy: TeichmullerColdDiffusionStrategy) -> torch.Tensor:
    grads = [
        p.grad.detach().flatten()
        for p in strategy.schedule_head.parameters()
        if p.grad is not None
    ]
    assert grads, "schedule_head received no gradient at all"
    return torch.cat(grads)


# ---------------------------------------------------------------------------
# F3 — missing typed sub-block must RAISE (no silent fallback).
# ---------------------------------------------------------------------------
def test_missing_teichmuller_block_raises() -> None:
    """Pre-fix: getattr defaults silently to 0.95/16/8. Post-fix: raises."""
    with pytest.raises(ValueError, match="training.teichmuller"):
        _build_strategy(_simple_model(), teichmuller=None)


# ---------------------------------------------------------------------------
# F4 — μ_T is spatially varying with |μ_T| < 1.
# ---------------------------------------------------------------------------
def test_mu_T_is_spatially_varying_inside_unit_ball() -> None:
    cfg = TeichmullerTrainingConfigSchema(spatial_size=16, mu_t_max=0.7)
    strategy = _build_strategy(_simple_model(), teichmuller=cfg)

    mu_T = strategy.muT
    assert mu_T.is_complex()
    mag = mu_T.abs()
    # |μ_T| strictly inside the unit ball (homeomorphism constraint).
    assert torch.all(mag < 1.0)
    assert float(mag.max()) <= 0.7 + 1e-5
    # NON-uniform: the degenerate old endpoint was a constant 0.5 map.
    assert float(mag.std()) > 1e-3, "μ_T endpoint is spatially uniform (degenerate)"


# ---------------------------------------------------------------------------
# F1/F2 — schedule head gets a non-zero, batch-dependent gradient.
# ---------------------------------------------------------------------------
def test_schedule_head_receives_nonzero_gradient() -> None:
    torch.manual_seed(0)
    cfg = TeichmullerTrainingConfigSchema(spatial_size=8, n_steps=6)
    strategy = _build_strategy(_simple_model(), teichmuller=cfg)

    strategy.schedule_head.zero_grad(set_to_none=True)
    target = torch.randn(2, 1, 16, 16)
    # Fix the random step so the result is deterministic for the assertion.
    torch.manual_seed(123)
    losses = strategy._compute_losses_impl(input_batch=None, target_batch=target)
    losses["g_total_loss"].backward()

    grad = _schedule_grad_vector(strategy)
    # Pre-fix the only schedule-coupled term was ``0.0 * μ`` (zero gradient)
    # plus the batch-independent ``teich_smooth``; the degrade→restore loss
    # now drives a genuine non-zero gradient.
    assert float(grad.abs().sum()) > 0.0


def test_schedule_head_gradient_is_batch_dependent() -> None:
    cfg = TeichmullerTrainingConfigSchema(spatial_size=8, n_steps=6)
    strategy = _build_strategy(_simple_model(), teichmuller=cfg)

    torch.manual_seed(7)
    batch_a = torch.randn(2, 1, 16, 16)
    torch.manual_seed(11)
    batch_b = torch.randn(2, 1, 16, 16) * 3.0 + 1.0

    # Seed identically before each loss so the SAME schedule step is sampled;
    # any difference in the gradient then comes purely from the data.
    strategy.schedule_head.zero_grad(set_to_none=True)
    torch.manual_seed(42)
    strategy._compute_losses_impl(
        input_batch=None, target_batch=batch_a
    )["g_total_loss"].backward()
    grad_a = _schedule_grad_vector(strategy)

    strategy.schedule_head.zero_grad(set_to_none=True)
    torch.manual_seed(42)
    strategy._compute_losses_impl(
        input_batch=None, target_batch=batch_b
    )["g_total_loss"].backward()
    grad_b = _schedule_grad_vector(strategy)

    # Pre-fix: the schedule-coupled term was data-independent (``teich_smooth``
    # depends only on r), so grad_a == grad_b. Post-fix the degrade→restore
    # loss makes the gradient depend on the batch.
    assert not torch.allclose(grad_a, grad_b, atol=1e-7)


# ---------------------------------------------------------------------------
# F3 (cont.) — resolved knobs stamped into provenance.
# ---------------------------------------------------------------------------
def test_resolved_config_is_stamped() -> None:
    cfg = TeichmullerTrainingConfigSchema(
        r_max=0.9, spatial_size=8, n_steps=4, lambda_teichmuller=0.02
    )
    strategy = _build_strategy(_simple_model(), teichmuller=cfg)
    stamped = strategy.resolved_teichmuller_config
    assert stamped["r_max"] == 0.9
    assert stamped["spatial_size"] == 8
    assert stamped["n_steps"] == 4
    assert stamped["lambda_teichmuller"] == 0.02


# ---------------------------------------------------------------------------
# Schema — unknown knob rejected (extra="forbid").
# ---------------------------------------------------------------------------
def test_schema_rejects_unknown_knob() -> None:
    with pytest.raises(Exception):
        TeichmullerTrainingConfigSchema(not_a_real_knob=1.0)


def test_schema_rejects_out_of_ball_r_max() -> None:
    with pytest.raises(Exception):
        TeichmullerTrainingConfigSchema(r_max=1.5)


# ---------------------------------------------------------------------------
# Owned state checkpoint round trip (#2629): the schedule head keeps its key.
# ---------------------------------------------------------------------------
def test_schedule_head_checkpoint_keys_and_round_trip() -> None:
    torch.manual_seed(0)
    cfg = TeichmullerTrainingConfigSchema(spatial_size=8, n_steps=6)
    saved = _build_strategy(_simple_model(), teichmuller=cfg)
    state = saved.strategy_state_dict()
    # Registration must not move the head out of the checkpoint's "modules" entry.
    assert set(state["modules"]) == {"schedule_head"}
    assert set(state["modules"]["schedule_head"]) == set(saved.schedule_head.state_dict())
    assert not any(k.startswith("module.") for k in state["modules"]["schedule_head"])

    torch.manual_seed(1)
    fresh = _build_strategy(_simple_model(), teichmuller=cfg)
    fresh.load_strategy_state_dict(state)
    for key, value in saved.schedule_head.state_dict().items():
        assert torch.equal(fresh.schedule_head.state_dict()[key], value), key
