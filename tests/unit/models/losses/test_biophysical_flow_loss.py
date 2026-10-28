"""BiophysicalFlowLoss reduces complex and real k-space residuals the same way."""

from __future__ import annotations

import torch

from spectramr.models.losses.biophysical_flow_loss import BiophysicalFlowLoss


def test_complex_dc_term_is_a_mean_not_a_sum() -> None:
    """``torch.norm(p=1)`` summed |z|, so a complex input scored N times its real twin."""
    torch.manual_seed(0)
    k_pred, k_target = torch.randn(1, 1, 32, 32), torch.randn(1, 1, 32, 32)
    loss_fn = BiophysicalFlowLoss()
    real = loss_fn(k_pred, k_target)["dc_loss"]
    as_complex = loss_fn(k_pred.to(torch.complex64), k_target.to(torch.complex64))["dc_loss"]
    torch.testing.assert_close(as_complex, real)


def test_complex_dc_term_is_the_mean_modulus() -> None:
    torch.manual_seed(1)
    k_pred = torch.randn(2, 1, 8, 8, dtype=torch.complex64)
    k_target = torch.randn(2, 1, 8, 8, dtype=torch.complex64)
    out = BiophysicalFlowLoss(lambda_dc=2.0)(k_pred, k_target)
    torch.testing.assert_close(out["dc_loss"], (k_pred - k_target).abs().mean())
    torch.testing.assert_close(out["total_loss"], 2.0 * out["dc_loss"])


def test_bloch_term_penalises_t2_above_t1() -> None:
    maps = torch.tensor([[[1.0, 100.0, 200.0]]])  # T2 > T1 is unphysical
    out = BiophysicalFlowLoss()(torch.zeros(1, 4), torch.zeros(1, 4), biophysics_map=maps)
    assert out["bloch_loss"].item() > 0.0
