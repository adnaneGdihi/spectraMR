"""Tests for :mod:`diffeomorphic_recon_strategy`: the warp terms are live and correct.

Pre-fix the three diffeomorphic terms were gated on ``batch['velocity_field']``, a key no
loader or model writes, so only the parent L1 trained. Scaling-and-squaring also
composed ``φ + φ∘φ - id`` (≈ 3u per squaring) with an extra 0.5 factor, and the fold
hinge read ``v`` rather than φ.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from spectramr.infrastructure.training.strategies.diffeomorphic_recon_strategy import (
    DiffeomorphicReconStrategy,
)
from spectramr.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)
from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

_ARM_STRATEGY_CLASS = (
    "spectramr.infrastructure.training.strategies.diffeomorphic_recon_strategy."
    "DiffeomorphicReconStrategy"
)


@pytest.mark.parametrize(
    "training",
    [
        SimpleNamespace(strategy_class=None, training_mode="diffeomorphic_recon"),
        SimpleNamespace(strategy_class=_ARM_STRATEGY_CLASS, training_mode="diffeomorphic_recon"),
    ],
)
def test_factory_resolves_the_strategy(training: SimpleNamespace) -> None:
    cls = TrainingStrategyFactory().get_strategy_class(SimpleNamespace(training=training))
    assert cls is DiffeomorphicReconStrategy


def _bare_strategy(opt: torch.optim.Optimizer | None = None) -> DiffeomorphicReconStrategy:
    strat = DiffeomorphicReconStrategy.__new__(DiffeomorphicReconStrategy)
    strat.config = SimpleNamespace(parallel=None)
    strat.env = SimpleNamespace(opt_g=opt)
    strat.device = None
    strat._owned_modules = {}
    strat._owned_wrappers = {}
    return strat


def test_constant_velocity_exponentiates_to_that_translation() -> None:
    strat = _bare_strategy()
    H = W = 16
    v = torch.zeros(1, 2, H, W)
    v[:, 0] = 2.0  # +2 px along x
    disp = strat.grid_displacement(strat.exp_velocity(v))
    interior = disp[:, :, 3:-3, 3:-5]
    assert torch.allclose(interior[:, 0], torch.full_like(interior[:, 0], 2.0), atol=1e-4)
    assert torch.allclose(interior[:, 1], torch.zeros_like(interior[:, 1]), atol=1e-4)


def test_exp_zero_is_identity() -> None:
    strat = _bare_strategy()
    disp = strat.grid_displacement(strat.exp_velocity(torch.zeros(2, 2, 12, 12)))
    assert disp.abs().max() < 1e-4


def test_fold_hinge_fires_on_a_folding_displacement() -> None:
    x = torch.arange(8.0).view(1, 1, 1, 8).expand(1, 1, 8, 8)
    flow = torch.cat([-2.0 * x, torch.zeros_like(x)], dim=1)  # x -> -x: orientation flips
    det = DiffeomorphicReconStrategy._jacobian_determinant(flow)
    assert det.shape == (1, 7, 7)
    assert torch.allclose(det, torch.full_like(det, -1.0))
    assert torch.relu(-det).mean() > 0


def _blob(shift: float, H: int = 16, W: int = 16) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij"
    )
    return torch.exp(-((xx - 7.5 - shift) ** 2 + (yy - 7.5) ** 2) / 8.0).view(1, 1, H, W)


def test_production_step_emits_v_and_trains_the_head(monkeypatch: pytest.MonkeyPatch) -> None:
    """The head built by the strategy is on opt_g, its warp terms reach g_total_loss,
    and optimising them registers a shifted target (the mechanism fires)."""
    gen = nn.Conv2d(1, 1, 1)
    nn.init.ones_(gen.weight)
    nn.init.zeros_(gen.bias)
    opt = torch.optim.Adam(gen.parameters(), lr=0.0)
    strat = _bare_strategy(opt)
    strat._build_velocity_head(1)
    head = strat.owned_modules()["velocity_head"]
    head_params = {id(p) for p in head.parameters()}
    assert any(id(p) in head_params for g in opt.param_groups for p in g["params"])
    for group in opt.param_groups[1:]:
        group["lr"] = 5e-3

    def _parent(self, **kw):
        pred = gen(kw["input_batch"])
        target = kw["target_batch"]
        total = (pred - target).abs().mean()
        self._last_prediction, self._last_target = pred, target
        return {"g_total_loss": total, "loss": total}

    monkeypatch.setattr(ReconstructionTrainingStrategy, "_compute_losses_impl", _parent)
    moving, target = _blob(0.0), _blob(1.5)

    first = strat._compute_losses_impl(input_batch=moving, target_batch=target, epoch=0)
    for key in (
        "loss_velocity_smoothness",
        "loss_diffeomorphism_folding",
        "loss_warped_fidelity",
        "diffeo_fold_fraction",
    ):
        assert key in first
    assert first["loss_warped_fidelity"] > 0
    parent_l1 = (moving - target).abs().mean()
    assert torch.allclose(
        first["g_total_loss"],
        parent_l1
        + first["loss_warped_fidelity"]
        + first["loss_velocity_smoothness"]
        + first["loss_diffeomorphism_folding"],
    )
    assert first["loss"] is first["g_total_loss"]
    first["g_total_loss"].backward()
    assert head.net[-1].weight.grad.abs().sum() > 0
    assert gen.weight.grad is not None
    start = float(first["loss_warped_fidelity"].detach())

    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        out = strat._compute_losses_impl(input_batch=moving, target_batch=target, epoch=0)
        out["g_total_loss"].backward()
        opt.step()
    assert float(out["loss_warped_fidelity"].detach()) < 0.5 * start
    assert float(out["diffeo_fold_fraction"]) == 0.0


def test_declared_metric_keys_carry_the_warp_columns() -> None:
    strat = _bare_strategy()
    strat.config = SimpleNamespace(
        parallel=None, model=SimpleNamespace(model_type="configurable_unet")
    )
    keys = strat.declared_metric_keys()
    assert {"loss_warped_fidelity", "diffeo_fold_fraction"} <= keys


def test_empty_seam_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    strat = _bare_strategy()
    strat._last_prediction = strat._last_target = None
    monkeypatch.setattr(
        ReconstructionTrainingStrategy,
        "_compute_losses_impl",
        lambda self, **kw: {"g_total_loss": torch.zeros(())},
    )
    with pytest.raises(RuntimeError, match="seam"):
        strat._compute_losses_impl(
            input_batch=torch.zeros(1, 1, 4, 4), target_batch=torch.zeros(1, 1, 4, 4)
        )


def test_applied_lddmm_constants_reach_the_snapshot_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ReconstructionTrainingStrategy, "_first_steps_extra", lambda self, epoch, batch=None: {}
    )
    lddmm = _bare_strategy()._first_steps_extra(0)["lddmm"]
    assert lddmm["n_squarings"] == DiffeomorphicReconStrategy.n_squarings
    assert lddmm["lambda_warp"] == DiffeomorphicReconStrategy.lambda_warp
    assert lddmm["velocity_source"].startswith("velocity_field_head")


def test_a_parent_step_without_a_total_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    strat = _bare_strategy()
    strat._last_prediction = strat._last_target = torch.zeros(1, 1, 4, 4)
    monkeypatch.setattr(
        ReconstructionTrainingStrategy, "_compute_losses_impl", lambda self, **kw: {}
    )
    monkeypatch.setattr(
        DiffeomorphicReconStrategy, "call_owned", lambda self, name, x: torch.zeros(1, 2, 4, 4)
    )
    with pytest.raises(RuntimeError, match="g_total_loss"):
        strat._compute_losses_impl(
            input_batch=torch.zeros(1, 1, 4, 4), target_batch=torch.zeros(1, 1, 4, 4)
        )
