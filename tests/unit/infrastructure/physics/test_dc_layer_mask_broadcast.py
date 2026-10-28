"""Regression tests for the DC-layer mask-broadcast guard.

The 2026-05-10 cluster rerun saw four ``experiment_11_diff_varnet*``
arms crash with::

    RuntimeError: The size of tensor a (8) must match the size of
                  tensor b (4) at non-singleton dimension 1

at ``data_consistency_layer.py``'s ``k * (1-mask) + measured * mask``
expression. The k-space tensor had 8 complex channels (4 coils × R/I)
but the mask handed in had 4 — neither broadcastable nor singleton.

Physics: the k-space sampling mask is coil-independent (the pattern is
the same across every coil), so when the channel counts disagree the
correct fix is to collapse the mask along its channel dim with a
logical-OR reduction (any coil sampled this freq → that freq is
"sampled" everywhere). This test pins that behavior.
"""
from __future__ import annotations

import torch

from spectramr.infrastructure.physics.data_consistency_layer import MaskedReplacementDataConsistency


def test_complex_input_broadcasts_full_mask():
    """Single-channel mask broadcasts cleanly without modification."""
    dc = MaskedReplacementDataConsistency()
    image = torch.randn(2, 4, 16, 16, dtype=torch.complex64)
    measured = torch.randn(2, 4, 16, 16, dtype=torch.complex64)
    mask = torch.zeros(2, 1, 16, 16)
    mask[..., 8, :] = 1
    out = dc(image, measured, mask)
    assert out.shape == image.shape


def test_complex_mismatched_mask_collapsed():
    """Mask with channels disagreeing with k-space gets collapsed via amax."""
    dc = MaskedReplacementDataConsistency()
    image = torch.randn(2, 8, 16, 16, dtype=torch.complex64)
    measured = torch.randn(2, 8, 16, 16, dtype=torch.complex64)
    # Mask has 4 channels — NOT broadcastable with 8-coil k-space.
    mask = torch.zeros(2, 4, 16, 16)
    # Sample one freq in coil 0, another in coil 2 — collapse should OR them
    mask[:, 0, 4, :] = 1
    mask[:, 2, 12, :] = 1
    out = dc(image, measured, mask)
    assert out.shape == image.shape


def test_real_stacked_mismatched_mask_collapsed():
    """Same guard works for the real-stacked-complex (interleaved) input path."""
    dc = MaskedReplacementDataConsistency()
    # 4-channel real-stacked = 2 complex coils
    image = torch.randn(2, 4, 16, 16)
    measured = torch.randn(2, 4, 16, 16)
    # Mismatch: mask has 3 channels which is neither 1 nor 2
    mask = torch.zeros(2, 3, 16, 16)
    mask[:, 1, 8, :] = 1
    out = dc(image, measured, mask)
    assert out.shape == image.shape


def test_bool_mask_converted_and_broadcast():
    """Boolean mask is converted to float and the broadcast rule still applies."""
    dc = MaskedReplacementDataConsistency()
    image = torch.randn(1, 4, 8, 8, dtype=torch.complex64)
    measured = torch.zeros_like(image)
    mask = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    mask[..., 4, :] = True
    out = dc(image, measured, mask)
    assert out.shape == image.shape
