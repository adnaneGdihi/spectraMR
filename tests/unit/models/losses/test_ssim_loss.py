"""Window knobs of ``SSIMLoss`` / ``MSSSIMLoss``.

Behavioural SSIM coverage lives in ``tests/unit/losses/test_ssim_loss.py``. This
file pins that each advertised window knob reaches the window: ``SSIMLoss`` took
``win_sigma`` and ``MSSSIMLoss`` took ``window_size``/``sigma`` and read none of
them, so a caller's value was dropped without a word.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.core.metrics.evaluation_metrics import gaussian_kernel
from spectramr.models.losses.ssim_loss import MSSSIMLoss, SSIMLoss

_CPU = torch.device("cpu")


def test_ssim_defaults_are_the_wang_window() -> None:
    """Default window is 11x11 with sigma 1.5, K1=0.01, K2=0.03."""
    loss = SSIMLoss()
    assert (loss.window_size, loss.sigma, loss.K1, loss.K2) == (11, 1.5, 0.01, 0.03)
    torch.testing.assert_close(loss.window, gaussian_kernel(11, 1.5, _CPU))


def test_ssim_win_sigma_reaches_the_window() -> None:
    """``win_sigma`` alone sets the Gaussian width."""
    loss = SSIMLoss(win_sigma=3.0)
    assert loss.sigma == 3.0
    torch.testing.assert_close(loss.window, gaussian_kernel(11, 3.0, _CPU))


def test_ssim_conflicting_sigma_spellings_raise() -> None:
    with pytest.raises(ValueError, match="win_sigma"):
        SSIMLoss(sigma=1.5, win_sigma=3.0)


def test_msssim_window_size_reaches_the_inner_ssim() -> None:
    """``window_size`` / ``sigma`` configure the per-scale SSIM window."""
    loss = MSSSIMLoss(window_size=7, sigma=1.0)
    assert (loss._ssim.window_size, loss._ssim.sigma) == (7, 1.0)
    torch.testing.assert_close(loss._ssim.window, gaussian_kernel(7, 1.0, _CPU))


def test_msssim_defaults_unchanged() -> None:
    loss = MSSSIMLoss()
    assert (loss._ssim.window_size, loss._ssim.sigma) == (11, 1.5)


def test_msssim_conflicting_window_spellings_raise() -> None:
    with pytest.raises(ValueError, match="win_size"):
        MSSSIMLoss(window_size=7, win_size=11)


def _reference_msssim(loss: MSSSIMLoss, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-scale SSIM with the exponents rebuilt from the Python list each call."""
    import torch.nn.functional as F

    dr = loss._ssim.resolve_data_range(pred, target)
    values = []
    a, b = pred, target
    for i in range(len(loss.weights)):
        values.append(loss._ssim.compute_ssim_with_resolved_range(a, b, dr))
        if i < len(loss.weights) - 1:
            a, b = F.avg_pool2d(a, 2, 2), F.avg_pool2d(b, 2, 2)
    stack = torch.stack(values).clamp(min=1e-6)
    exps = torch.tensor(loss.weights, dtype=pred.dtype).unsqueeze(1)
    return (1.0 - torch.prod(torch.pow(stack, exps), dim=0)).mean()


def test_msssim_buffered_exponents_match_the_list() -> None:
    gen = torch.Generator().manual_seed(0)
    target = torch.rand(1, 1, 64, 64, generator=gen)
    pred = (target + 0.1 * torch.randn(1, 1, 64, 64, generator=gen)).clamp(0, 1)
    loss = MSSSIMLoss()
    torch.testing.assert_close(loss(pred, target), _reference_msssim(loss, pred, target))
    assert "scale_weights" not in loss.state_dict()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sync detection needs CUDA")
def test_msssim_forward_does_not_synchronise_the_device() -> None:
    loss = MSSSIMLoss().cuda()
    pred = torch.rand(1, 1, 96, 96, device="cuda")
    target = torch.rand(1, 1, 96, 96, device="cuda")
    loss(pred, target)
    torch.cuda.set_sync_debug_mode("error")
    try:
        loss(pred, target)
    finally:
        torch.cuda.set_sync_debug_mode("default")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a second device")
def test_msssim_left_on_cpu_still_scores_a_cuda_batch() -> None:
    """The per-call window move made a CPU-resident module device-agnostic; the
    buffered exponents keep that contract."""
    pred = torch.rand(1, 1, 64, 64, device="cuda")
    assert torch.isfinite(MSSSIMLoss()(pred, pred.clone()))
