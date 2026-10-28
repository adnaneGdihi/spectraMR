"""End-to-end registration + smoke tests for the SFC / conformal
components introduced by ``TODO/audit_plan_novel.md`` SFC plan.

Verifies (i) every advertised name is in the corresponding registry,
(ii) each component performs one forward pass without error.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

# Force-populate registries.
import spectramr.models.generators  # noqa: F401
import spectramr.models.diffusion  # noqa: F401
import spectramr.models.losses  # noqa: F401


def test_conformal_geomamba_ulf_registered_and_runs() -> None:
    from spectramr.models.registry import MODEL_REGISTRY
    assert "conformal_geomamba_ulf" in MODEL_REGISTRY
    from spectramr.models.generators.conformal_geomamba_ulf import ConformalGeoMambaULF
    model = ConformalGeoMambaULF(
        in_channels=1, out_channels=1, base_width=8, scales=2, grid_size=16
    )
    out = model(torch.randn(2, 1, 64, 64))
    assert out.shape == (2, 1, 64, 64)


def test_three_conformal_losses_registered() -> None:
    from spectramr.models.losses.registry import LossRegistry
    names = set(LossRegistry.list_available()) | set(LossRegistry._custom_losses)
    for n in ("conformal_modulus", "beltrami_motion_residual", "teichmuller_geodesic"):
        assert n in names, f"loss {n!r} not registered"


def test_beltrami_sfc_block_returns_valid_permutation() -> None:
    from spectramr.models.blocks.space_filling_curves import BeltramiSFCBlock
    blk = BeltramiSFCBlock(grid_size=8, in_channels=1)
    out = blk(torch.randn(2, 1, 8, 8))
    assert out["indices"].shape == (2, 64)
    # Each sample's indices is a permutation of [0, 64)
    for b in range(2):
        assert out["indices"][b].unique().numel() == 64
    assert out["mu"].abs().amax().item() <= 0.9


def test_beltrami_sfc_block_zero_mu_is_hilbert() -> None:
    """μ ≡ 0 ⇒ the adaptive ordering is EXACTLY the fixed Hilbert curve.

    The block warps the base Hilbert curve and orders cells by the Hilbert rank
    of their warped position; with μ=0 the warp is the identity, so the order is
    unchanged. This is the consistency claim the module docstring makes — the
    old ``φ.angle()`` sort did NOT satisfy it (it collapsed 2-D locality even at
    μ=0), and no test ever pinned it (F3)."""
    from spectramr.models.blocks.space_filling_curves import BeltramiSFCBlock

    gs = 16
    blk = BeltramiSFCBlock(grid_size=gs, in_channels=1)
    with torch.no_grad():  # force μ ≡ 0 by zeroing the head's last layer
        blk.mu_head[-1].weight.zero_()
        blk.mu_head[-1].bias.zero_()
    out = blk(torch.randn(3, 1, gs, gs))
    assert out["mu"].abs().amax().item() == 0.0
    for b in range(3):
        assert torch.equal(out["indices"][b], blk.hilbert_indices)


def test_beltrami_sfc_block_is_locality_preserving() -> None:
    """The warped adaptive curve keeps a bounded max L1 step between consecutive
    tokens — unlike the old global-angle sort (max step ≈ 13 on a 16×16 grid)."""
    from spectramr.models.blocks.space_filling_curves import BeltramiSFCBlock

    gs = 16
    torch.manual_seed(0)
    blk = BeltramiSFCBlock(grid_size=gs, in_channels=1)
    perm = blk(torch.randn(1, 1, gs, gs))["indices"][0]
    rc = torch.stack([perm // gs, perm % gs], dim=-1).float()
    max_step = (rc[1:] - rc[:-1]).abs().sum(-1).max().item()
    assert max_step <= 4.0  # Hilbert baseline is 1; the old angle-sort was ~13


def test_conformal_data_consistency_acts_as_dc_when_jac_one() -> None:
    from spectramr.infrastructure.physics.data_consistency import ConformalDataConsistency
    dc = ConformalDataConsistency(alpha=1.0)
    x = torch.randn(2, 2, 16, 16)
    y = torch.randn(2, 2, 16, 16)
    mask = torch.ones(2, 1, 16, 16)
    jac = torch.ones(2, 1, 16, 16)
    out = dc(x, y_tilde=y, mask=mask, conformal_jacobian=jac)
    assert torch.allclose(out, y, atol=1e-5)


def test_teichmuller_schedule_head_in_unit_interval() -> None:
    from spectramr.models.diffusion.teichmuller_schedule import TeichmullerScheduleHead
    head = TeichmullerScheduleHead(hidden=8, r_max=0.95)
    r = head(torch.linspace(0, 1, 16))
    assert r.shape == (16,)
    assert r.min().item() >= 0.0
    assert r.max().item() <= 0.95


def test_four_sfc_conformal_strategies_registered() -> None:
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory
    sm = TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    for n in (
        "adaptive_sfc_hssc",
        "conformal_diffusion_recon",
        "beltrami_motion_correction",
        "cortical_conformal_recon",
    ):
        assert n in sm, f"strategy {n!r} not registered"


# --- Strategy smoke tests (bypass BaseTrainingStrategy.__init__) -----------


def _patched(strategy_cls, *, generator: nn.Module):
    """Build a strategy without the full DI bootstrap, mirroring the
    pattern used by tests/unit/strategies/test_novel_2026_strategies_smoke.py."""
    inst = strategy_cls.__new__(strategy_cls)
    inst.config = SimpleNamespace(training=SimpleNamespace(seed=0))
    inst.device = torch.device("cpu")
    inst.env = SimpleNamespace(
        models={"generator": generator},
        device=torch.device("cpu"),
        config=inst.config,
    )
    # generator_model is a property on BaseTrainingStrategy; install one
    # that resolves via env so all four strategies receive their model.
    type(inst).generator_model = property(  # type: ignore[assignment]
        lambda s: s.env.models["generator"]
    )
    return inst


class _ConstantModel(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


def test_adaptive_sfc_hssc_strategy_smoke() -> None:
    from spectramr.infrastructure.training.strategies.sfc_conformal_kspace_strategies import (
        AdaptiveSFCHSSCStrategy,
    )
    strat = _patched(AdaptiveSFCHSSCStrategy, generator=_ConstantModel())
    AdaptiveSFCHSSCStrategy.__init__.__wrapped__ if False else None  # noqa
    # Manually mirror init side-effects we need.
    from spectramr.models.blocks.space_filling_curves import BeltramiSFCBlock
    strat.sfc = BeltramiSFCBlock(grid_size=16, in_channels=1)
    # What register_owned_module records in __init__ (the call_owned lookup reads it).
    strat._owned_modules = {"sfc": strat.sfc}
    strat._owned_wrappers = {}
    strat.lambda_mu = 0.01
    strat.lambda_traj = 0.01
    x = torch.randn(1, 1, 32, 32)
    out = strat._compute_losses_impl(x, x, epoch=0)
    assert "g_total_loss" in out
    assert torch.isfinite(out["g_total_loss"])


def test_conformal_diffusion_recon_strategy_smoke() -> None:
    from spectramr.infrastructure.training.strategies.sfc_conformal_kspace_strategies import (
        ConformalDiffusionReconStrategy,
    )
    from spectramr.infrastructure.physics.data_consistency import ConformalDataConsistency
    strat = _patched(ConformalDiffusionReconStrategy, generator=_ConstantModel())
    strat.sigma_min = 0.01
    strat.sigma_max = 0.5
    strat.dc = ConformalDataConsistency(alpha=1.0)
    x = torch.randn(1, 1, 16, 16)
    # The strategy fails loud without the conformal-DC keys (pitfall #16); the
    # keys are populated by SFCConformalFMRIKeysWrapper in a real run. Provide
    # them here so the smoke exercises the actual conformal DC path.
    batch = {
        "mask": torch.ones(1, 1, 16, 16),
        "conformal_jacobian": torch.ones(1, 1, 16, 16),
    }
    out = strat._compute_losses_impl(x, x, epoch=0, batch=batch)
    assert torch.isfinite(out["g_total_loss"])


def test_conformal_diffusion_recon_strategy_raises_without_keys() -> None:
    """Absent ``mask`` / ``conformal_jacobian`` the strategy must RAISE, not
    silently run a vanilla denoising MSE (pitfall #16)."""
    import pytest

    from spectramr.infrastructure.physics.data_consistency import ConformalDataConsistency
    from spectramr.infrastructure.training.strategies.sfc_conformal_kspace_strategies import (
        ConformalDiffusionReconStrategy,
    )

    strat = _patched(ConformalDiffusionReconStrategy, generator=_ConstantModel())
    strat.sigma_min = 0.01
    strat.sigma_max = 0.5
    strat.dc = ConformalDataConsistency(alpha=1.0)
    x = torch.randn(1, 1, 16, 16)
    with pytest.raises(ValueError, match="requires 'mask' and 'conformal_jacobian'"):
        strat._compute_losses_impl(x, x, epoch=0)


def test_beltrami_motion_correction_strategy_smoke() -> None:
    from spectramr.infrastructure.training.strategies.beltrami_motion_cortical_strategies import (
        BeltramiMotionCorrectionStrategy,
    )
    from spectramr.infrastructure.physics.conformal_geometry import LinearBeltramiSolver
    strat = _patched(BeltramiMotionCorrectionStrategy, generator=_ConstantModel())
    strat.grid_size = 16
    strat.k_max = 0.9
    strat.lambda_smooth = 1e-3
    strat.solver = LinearBeltramiSolver(max_iter=10, tol=1e-3)
    strat.mu_head = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.GELU(), nn.Conv2d(8, 2, 3, padding=1)
    )
    strat._owned_modules = {"mu_head": strat.mu_head}
    strat._owned_wrappers = {}
    x = torch.randn(1, 1, 32, 32)
    out = strat._compute_losses_impl(x, x, epoch=0)
    assert torch.isfinite(out["g_total_loss"])


def test_cortical_conformal_recon_strategy_smoke() -> None:
    from spectramr.infrastructure.training.strategies.beltrami_motion_cortical_strategies import (
        CorticalConformalReconStrategy,
    )
    strat = _patched(CorticalConformalReconStrategy, generator=_ConstantModel())
    strat.lambda_curv = 0.1
    x = torch.randn(1, 1, 16, 16)
    out = strat._compute_losses_impl(x, x, epoch=0)
    assert torch.isfinite(out["g_total_loss"])
