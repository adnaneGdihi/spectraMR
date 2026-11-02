"""Regression tests for the 2026-06 slice_to_volume_strategy audit fixes.

Covers the two BROKEN_STRATEGY_ONLY findings against
``SliceToVolumeStrategy._through_plane_consistency``:

1. ``kernel < 1`` (i.e. ``D_hi < D_lo``) used to silently swap the
   consistency operator to trilinear interpolation; it must now raise
   because a slice-to-volume upsampler MUST emit a denser through-plane
   output than its input (pitfall #9, no silent fallback).
2. A non-integer through-plane ratio (``D_hi % D_lo != 0``) used to fall
   through to a trilinear-resize fallback, silently switching the
   consistency operator depending on shape; it must now raise so that
   average-pool is the only consistency operator.

The method does not touch ``self`` on the tested path, so it is invoked on
a bare instance built via ``object.__new__`` to keep the fakes tiny.
"""
from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.training.strategies.slice_to_volume_strategy import (
    SliceToVolumeStrategy,
)


def _bare_strategy() -> SliceToVolumeStrategy:
    """A SliceToVolumeStrategy with no ``__init__`` run (path uses no self)."""
    return object.__new__(SliceToVolumeStrategy)


def test_raises_when_output_less_dense_than_input() -> None:
    """D_hi < D_lo is a contract violation and must raise, not interpolate."""
    strat = _bare_strategy()
    x_hi = torch.zeros(1, 1, 4, 8, 8)   # 4 through-plane samples
    x_lo = torch.zeros(1, 1, 8, 8, 8)   # 8 through-plane samples (denser input)
    with pytest.raises(ValueError, match="must be denser than input"):
        strat._through_plane_consistency(x_hi, x_lo)


def test_raises_on_non_integer_through_plane_ratio() -> None:
    """A non-integer D_hi/D_lo ratio must raise instead of silently resizing."""
    strat = _bare_strategy()
    x_hi = torch.zeros(1, 1, 10, 8, 8)  # 10 not divisible by 4
    x_lo = torch.zeros(1, 1, 4, 8, 8)
    with pytest.raises(ValueError, match="must be an integer"):
        strat._through_plane_consistency(x_hi, x_lo)


def test_integer_ratio_uses_avg_pool_only() -> None:
    """A clean integer ratio pools to exactly D_lo and returns a finite MSE."""
    strat = _bare_strategy()
    # D_hi = 8, D_lo = 2 -> kernel 4, avg_pool3d yields exactly 2 slices.
    x_hi = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8, 1, 1).expand(1, 1, 8, 4, 4).contiguous()
    x_lo = torch.zeros(1, 1, 2, 4, 4)
    loss = strat._through_plane_consistency(x_hi, x_lo)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    # avg of [0,1,2,3]=1.5 and [4,5,6,7]=5.5 vs zeros -> mean of squares.
    expected = ((1.5 ** 2) + (5.5 ** 2)) / 2.0
    assert loss.item() == pytest.approx(expected, rel=1e-5)


def test_equal_depth_without_a_decimation_stamp_raises() -> None:
    """D_hi == D_lo with no slice_profile stamp would be an identity MSE onto the input."""
    strat = _bare_strategy()
    x_hi = torch.ones(1, 1, 4, 4, 4)
    x_lo = torch.zeros(1, 1, 4, 4, 4)
    with pytest.raises(ValueError, match="no slice_profile decimation stamp"):
        strat._through_plane_consistency(x_hi, x_lo)


def test_non_5d_inputs_still_raise() -> None:
    """The pre-existing 5-D guard must remain intact alongside the new raises."""
    strat = _bare_strategy()
    with pytest.raises(ValueError, match="expects 5-D tensors"):
        strat._through_plane_consistency(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4))


def test_no_silent_interpolation_fallback_in_source() -> None:
    """The trilinear silent-fallback CODE was removed from the method.

    The fix deletes both ``F.interpolate(...)`` resize calls (the
    ``kernel < 1`` branch and the post-pool depth-mismatch crop) so that
    average-pool is the only consistency operator. The words may still
    appear in explanatory comments; we assert the executable calls are gone.
    """
    import inspect

    src = inspect.getsource(SliceToVolumeStrategy._through_plane_consistency)
    assert "F.interpolate" not in src, "no resize fallback call should remain"
    assert "avg_pool3d" in src, "average-pool must be the consistency operator"


# --- Design item D3: gradient-isotropy prior replaces the mean-projection term ---


def _ramp_volume(dz: float, dh: float, dw: float, n: int = 6) -> torch.Tensor:
    """Volume x[d,h,w] = dz*d + dh*h + dw*w, so the per-axis mean |gradient| is
    exactly (dz, dh, dw). Lets us dial each axis's texture energy precisely."""
    dd = torch.arange(n).view(1, 1, n, 1, 1)
    hh = torch.arange(n).view(1, 1, 1, n, 1)
    ww = torch.arange(n).view(1, 1, 1, 1, n)
    return (dz * dd + dh * hh + dw * ww).float()


def test_isotropy_prior_zero_when_gradient_energy_matches() -> None:
    """Equal per-axis gradient energy -> the prior is ~0 (isotropic volume)."""
    strat = _bare_strategy()
    x = _ramp_volume(1.0, 1.0, 1.0)  # gz = gh = gw = 1
    loss = strat._isotropy_consistency(x)
    assert loss.dim() == 0
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_isotropy_prior_penalises_through_plane_blur() -> None:
    """A volume that is constant through-plane (gz=0) but sharp in-plane (gh=gw=1)
    is the SVR failure mode and must incur a positive penalty ~ in_plane^2."""
    strat = _bare_strategy()
    x = _ramp_volume(0.0, 1.0, 1.0)  # gz = 0, in-plane energy = 1
    loss = strat._isotropy_consistency(x)
    assert loss.item() == pytest.approx(1.0, rel=1e-5)  # (0 - 1)^2
    # And it is strictly worse than the isotropic case.
    assert loss.item() > strat._isotropy_consistency(_ramp_volume(1.0, 1.0, 1.0)).item()


def test_isotropy_prior_constant_volume_is_zero() -> None:
    """A constant volume has no gradients on any axis -> 0 (recon loss, not this
    term, is what forbids the trivial constant solution)."""
    strat = _bare_strategy()
    loss = strat._isotropy_consistency(torch.full((1, 1, 4, 4, 4), 3.0))
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_isotropy_prior_requires_5d() -> None:
    strat = _bare_strategy()
    with pytest.raises(ValueError, match="5-D"):
        strat._isotropy_consistency(torch.zeros(1, 1, 8, 8))


def test_old_mean_projection_term_is_gone() -> None:
    """The degenerate mean-projection cross-MSE must no longer exist."""
    import inspect

    assert not hasattr(SliceToVolumeStrategy, "_orthogonal_reformat_consistency")
    src = inspect.getsource(SliceToVolumeStrategy._isotropy_consistency)
    # No mean-projection along axes, no interpolate-to-common-grid.
    assert ".mean(dim=2)" not in src and ".mean(dim=3)" not in src
    assert "F.interpolate" not in src
