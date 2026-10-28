"""Grad-flow unit tests for HyperelasticJacobianLoss.

The canonical paired test file was missing (the review noted it was only
exercised functionally in test_vf_methods.py with no ``.backward()``). Pin that
the Jacobian-determinant regulariser is genuinely differentiable — a loss whose
gradient is zero / detached is inert regardless of the returned scalar.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.models.losses.hyperelastic_jacobian_loss import (  # noqa: E402
    HyperelasticJacobianLoss,
)
from spectramr.models.losses.registry import list_available  # noqa: E402


def test_registered() -> None:
    assert "hyperelastic_jacobian" in list_available()


def test_2d_deformation_field_grad_flows() -> None:
    loss = HyperelasticJacobianLoss()
    # A 2-D deformation field carries an (x, y) displacement per voxel.
    field = torch.randn(1, 2, 12, 12, requires_grad=True)
    out = loss(field)
    assert out.shape == ()
    out.backward()
    assert field.grad is not None
    assert torch.any(field.grad != 0)


def test_3d_deformation_field_grad_flows() -> None:
    loss = HyperelasticJacobianLoss()
    field = torch.randn(1, 3, 6, 6, 6, requires_grad=True)
    out = loss(field)
    out.backward()
    assert field.grad is not None
    assert torch.any(field.grad != 0)


def test_intermediate_outputs_deformation_field_is_used() -> None:
    """When a deformation field is supplied via ``intermediate_outputs`` it is
    used (not the prediction), and gradient reaches it."""
    loss = HyperelasticJacobianLoss()
    pred = torch.randn(1, 1, 12, 12)  # 1-channel image, NOT a field
    field = torch.randn(1, 2, 12, 12, requires_grad=True)
    out = loss(pred, intermediate_outputs=[field])
    out.backward()
    assert field.grad is not None
    assert torch.any(field.grad != 0)


def test_single_channel_input_raises_instead_of_returning_nan() -> None:
    """A C=1 input is a contract violation and must RAISE, not report nan.

    Regression for cluster job 8004252. ``_jacobian_det_2d`` slices
    ``u[:, 1:2]`` for the y-displacement, so on a 1-channel image that slice
    is EMPTY, every derivative downstream is empty, and ``torch.mean`` of an
    empty tensor is nan. The loss returned that nan as its value with a zero
    gradient -- reported into the total, contributing nothing to the update
    (pitfall #16) -- and ``tests/fuzz/loss_composition_fuzz`` independently
    reached a non-finite composed total through this path on shape
    ``(1, 1, 8, 8)``.

    The pre-existing guard checked ``u.dim()`` only. Rank was never the axis
    that broke; the channel count was.
    """
    loss = HyperelasticJacobianLoss()
    with pytest.raises(ValueError, match="channels"):
        loss(torch.randn(1, 1, 12, 12))


def test_3d_field_with_two_channels_raises() -> None:
    """A 5-D field needs 3 displacement channels, not 2."""
    loss = HyperelasticJacobianLoss()
    with pytest.raises(ValueError, match="channels"):
        loss(torch.randn(1, 2, 6, 6, 6))


def test_wrong_rank_still_raises_on_rank() -> None:
    """The pre-existing rank guard is kept, not replaced by the channel one."""
    loss = HyperelasticJacobianLoss()
    with pytest.raises(ValueError, match="4-D or 5-D"):
        loss(torch.randn(2, 12))


def test_3d_determinant_matches_a_closed_form_compression() -> None:
    """u = (0, 0, -z/2) halves volume along z: det J = 0.5 on the interior.

    Pins the per-axis derivative loop: each iteration must difference a different
    spatial axis, and channel 2 (u_z) must be read against axis 2 (depth).
    """
    d = h = w = 6
    zz = torch.arange(d, dtype=torch.float64).view(1, d, 1, 1).expand(1, d, h, w)
    field = torch.zeros(1, 3, d, h, w, dtype=torch.float64)
    field[:, 2] = -0.5 * zz
    det = HyperelasticJacobianLoss._jacobian_det_3d(field)
    torch.testing.assert_close(det, torch.full_like(det, 0.5))
