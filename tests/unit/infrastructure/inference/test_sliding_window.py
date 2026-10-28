"""Tests for :mod:`spectramr.infrastructure.inference.sliding_window`."""

from __future__ import annotations

import torch

from spectramr.infrastructure.inference.sliding_window import sliding_window_inference


def test_a_stitched_pointwise_prediction_equals_the_whole_volume_for_every_batch_row():
    """Patches are stacked along dim 0 with the whole batch each; reading one row per
    patch put batch row 0's prediction into every row once two volumes were stacked."""
    torch.manual_seed(0)
    x = torch.randn(2, 1, 12, 10, 9)
    out = sliding_window_inference(x, (6, 6, 5), lambda p: 2 * p, sw_batch_size=3)
    torch.testing.assert_close(out, 2 * x)


def test_a_volume_smaller_than_the_window_is_padded_and_cropped_back():
    """Reflection cannot pad an axis by more than its length; the window still runs."""
    x = torch.randn(1, 1, 3, 8, 8)
    out = sliding_window_inference(x, (8, 8, 8), lambda p: p + 1)
    torch.testing.assert_close(out, x + 1)


def test_the_2d_variant_covers_the_tail_and_pads_a_small_image():
    from spectramr.infrastructure.inference.sliding_window import sliding_window_inference_2d

    x = torch.randn(2, 1, 12, 7)
    out = sliding_window_inference_2d(x, (6, 8), lambda p: 2 * p)
    torch.testing.assert_close(out, 2 * x)
