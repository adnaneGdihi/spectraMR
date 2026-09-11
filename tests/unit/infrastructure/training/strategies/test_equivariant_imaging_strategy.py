r"""Tests for the Equivariant Imaging (EI) reconstruction strategy (Phase A step 3).

EI (Chen et al., ICCV 2021) supervises a reconstruction network *without ground
truth* by enforcing :math:`f(A T_g f(A^* y)) = T_g f(A^* y)`. The registered
``EquivariantSSLReconLoss`` was an **inert facade** — nothing emitted
``context["transformed_recon"]`` (the reference branch :math:`T_g f(A^* y)`), so
the term could never fire. These tests pin the *mechanism-fires* contract: the
pure core ``equivariant_imaging_branches`` produces both branches, and feeding
them to the registered loss yields a non-zero, grad-carrying objective that
actually pushes gradient into the reconstruction network.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.infrastructure.physics.group_actions import DihedralGroup
from spectramr.infrastructure.training.strategies.equivariant_imaging_strategy import (
    EquivariantImagingStrategy,
    equivariant_imaging_branches,
)
from spectramr.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)
from spectramr.models.losses.equivariant_recon_loss import EquivariantSSLReconLoss


def _partial_mask(h: int, w: int):
    mask = torch.zeros(1, 1, h, w)
    mask[..., :, ::4] = 1.0
    mask[..., :, w // 2 - 2 : w // 2 + 2] = 1.0
    return mask


def _ei_closures(h: int = 16, w: int = 16):
    """Build (reconstruct, forward_op, measured_kspace, net) for a tiny EI loop."""
    torch.manual_seed(0)
    mask_c = _partial_mask(h, w).to(torch.complex64)
    net = nn.Conv2d(1, 1, 3, padding=1)

    def forward_op(x):
        x_c = x if x.is_complex() else torch.complex(x, torch.zeros_like(x))
        return mask_c * fft2c(x_c)

    def reconstruct(kspace):
        zero_filled = ifft2c(mask_c * kspace).real  # A^* y (real)
        return net(zero_filled)

    phantom = torch.randn(1, 1, h, w)
    y = forward_op(phantom)
    return reconstruct, forward_op, y, net


# ---------------------------------------------------------------------------
# Pure core — branches + mechanism-fires
# ---------------------------------------------------------------------------
def test_branches_shapes_and_grad():
    reconstruct, forward_op, y, _net = _ei_closures()
    group = DihedralGroup()
    x_hat, pred, transformed = equivariant_imaging_branches(
        reconstruct, forward_op, group, y, g_index=2
    )
    assert x_hat.shape == pred.shape == transformed.shape
    assert pred.requires_grad and transformed.requires_grad


def test_identity_element_leaves_reference_equal_to_recon():
    """g == 0 (identity) => transformed_recon == x_hat exactly (clean invariant)."""
    reconstruct, forward_op, y, _net = _ei_closures()
    group = DihedralGroup()
    x_hat, _pred, transformed = equivariant_imaging_branches(
        reconstruct, forward_op, group, y, g_index=0
    )
    assert torch.equal(transformed, x_hat)


def test_mechanism_fires_equivariance_loss_trains_the_network():
    """The whole point of the un-trap: the registered EI loss must produce a
    non-zero, grad-carrying objective whose gradient reaches the recon net."""
    reconstruct, forward_op, y, net = _ei_closures()
    group = DihedralGroup()
    _x_hat, pred, transformed = equivariant_imaging_branches(
        reconstruct, forward_op, group, y, g_index=3
    )
    loss = EquivariantSSLReconLoss()(
        prediction=pred,
        target=torch.zeros_like(pred),
        context={"transformed_recon": transformed},
    )
    assert loss.ndim == 0 and loss.requires_grad
    assert float(loss.detach()) > 0.0, "equivariance loss is zero — mechanism did not fire"
    net.zero_grad(set_to_none=True)
    loss.backward()
    grad_sq = sum(float(p.grad.pow(2).sum()) for p in net.parameters() if p.grad is not None)
    assert grad_sq > 0.0, "EI loss produced no gradient on the recon network"


def test_branches_call_reconstruct_twice():
    """EI requires a second forward on the re-measured transformed recon."""
    reconstruct, forward_op, y, _net = _ei_closures()
    calls = {"n": 0}

    def counting_reconstruct(kspace):
        calls["n"] += 1
        return reconstruct(kspace)

    equivariant_imaging_branches(counting_reconstruct, forward_op, DihedralGroup(), y, g_index=1)
    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# Strategy class wiring
# ---------------------------------------------------------------------------
def test_strategy_extends_reconstruction():
    assert issubclass(EquivariantImagingStrategy, ReconstructionTrainingStrategy)


def test_strategy_routes_equivariance_through_registered_loss():
    """The strategy must consume the registered EquivariantSSLReconLoss (the
    facade target), not a private re-implementation — asserted from source."""
    import inspect

    src = inspect.getsource(EquivariantImagingStrategy)
    assert "transformed_recon" in src
    assert "EquivariantSSLReconLoss" in src
    # measurement-consistency anchor is present (prevents the trivial f≡const).
    assert "consistency" in src.lower()


def test_it_declares_its_own_loss_ownership() -> None:
    """Issue #1918: the objective is self-supervised: no ground truth is used at all.

    Read off ``__dict__``, never the inherited value: this class sits under
    ``ReconstructionTrainingStrategy``, whose ``folds_image_losses = True`` is truthful for ITSELF and
    becomes a lie the moment a subclass replaces ``_compute_losses_impl``. An
    inherited True makes the audit's ``image_losses_reach_the_objective`` witness
    PASS every declared ``losses.image_losses`` entry on this strategy's arms
    while the training step discards them.
    """
    assert EquivariantImagingStrategy.__dict__["folds_image_losses"] is False
    assert EquivariantImagingStrategy.__dict__["inline_losses"] == frozenset()
