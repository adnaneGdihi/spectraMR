"""``PersistentHomologyLoss`` range squash taken on device.

The autograd regression lives in ``test_topological_loss_autograd_fix.py``.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.topological_loss import PersistentHomologyLoss


def _reference(
    loss_fn: PersistentHomologyLoss, pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """The host-branching form: squash ``pred`` only when it leaves [0, 1]."""
    if pred.max() > 1.0 or pred.min() < 0.0:
        pred = torch.sigmoid(pred)
    loss = torch.zeros((), dtype=pred.dtype)
    for t in torch.linspace(0.1, 0.9, loss_fn.levels):
        ec_p = loss_fn.compute_euler_characteristic_2d(torch.sigmoid(20.0 * (pred - t)))
        ec_t = loss_fn.compute_euler_characteristic_2d(torch.sigmoid(20.0 * (target - t)))
        loss = loss + (ec_p - ec_t).pow(2).mean()
    return loss / loss_fn.levels


@pytest.mark.parametrize("offset", [0.0, 0.5, -0.5], ids=["in_range", "above", "below"])
def test_device_side_squash_matches_the_host_branch(offset: float) -> None:
    gen = torch.Generator().manual_seed(0)
    target = torch.rand(2, 1, 12, 12, generator=gen)
    pred = (torch.rand(2, 1, 12, 12, generator=gen) * 0.9 + offset).requires_grad_(True)
    loss_fn = PersistentHomologyLoss(levels=5)
    out = loss_fn(pred, target)
    (grad,) = torch.autograd.grad(out, pred)
    ref_pred = pred.detach().clone().requires_grad_(True)
    ref = _reference(loss_fn, ref_pred, target)
    (ref_grad,) = torch.autograd.grad(ref, ref_pred)
    torch.testing.assert_close(out, ref)
    torch.testing.assert_close(grad, ref_grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sync detection needs CUDA")
def test_forward_does_not_synchronise_the_device() -> None:
    pred = torch.rand(1, 1, 16, 16, device="cuda") + 0.5
    target = torch.rand(1, 1, 16, 16, device="cuda")
    loss_fn = PersistentHomologyLoss(levels=3)
    torch.cuda.set_sync_debug_mode("error")
    try:
        loss_fn(pred, target)
    finally:
        torch.cuda.set_sync_debug_mode("default")
