"""Kernel caching in ``HighFrequencyResidualLoss``.

Behavioural coverage (identity, attenuation, registry) lives in
``tests/unit/losses/test_high_frequency_residual_loss.py``; this file pins that
the Gaussian built once in ``__init__`` reproduces the per-call kernel exactly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from spectramr.models.losses.high_frequency_residual_loss import (
    HighFrequencyResidualLoss,
    _gaussian_lowpass,
)


def test_cached_kernel_matches_the_per_call_lowpass() -> None:
    """The buffered 1-D kernel gives the same loss as rebuilding it per call."""
    gen = torch.Generator().manual_seed(0)
    pred = torch.rand(2, 3, 24, 24, generator=gen)
    target = torch.rand(2, 3, 24, 24, generator=gen)
    loss_fn = HighFrequencyResidualLoss(sigma=1.5, truncate=2.5)
    pred_hf = pred - _gaussian_lowpass(pred, sigma=1.5, truncate=2.5)
    target_hf = target - _gaussian_lowpass(target, sigma=1.5, truncate=2.5)
    torch.testing.assert_close(loss_fn(pred, target), F.l1_loss(pred_hf, target_hf))


def test_cached_kernel_is_a_non_persistent_buffer_that_follows_dtype() -> None:
    """``.double()`` moves the kernel; the checkpoint layout is unchanged."""
    loss_fn = HighFrequencyResidualLoss(sigma=2.0).double()
    assert loss_fn.kernel_1d.dtype == torch.float64
    assert loss_fn.kernel_1d.shape == (1, 1, 13)
    assert "kernel_1d" not in loss_fn.state_dict()
    x = torch.rand(1, 1, 16, 16, dtype=torch.float64)
    assert loss_fn(x, x).item() == 0.0
