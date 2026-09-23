"""The calibration order that ``estimate_smaps_calibrated`` owns (#2213).

Three runtime consumers -- training, validation and sampling -- each held their
own copy of "estimate, normalise, match resolution", and all three ran it in the
order that breaks: normalise RSS on the ``acs_size`` grid, then bilinearly
interpolate ``real`` and ``imag`` separately up to the image size. Linear
interpolation between two unit-modulus complex numbers whose phases differ
returns the chord rather than the arc, so the modulus collapses between grid
nodes.

Every test below that ends ``_violation`` builds the broken order explicitly and
asserts it *fails*, so the assertion that the owner holds is measured against a
shape observed to break it rather than against nothing (non-negotiable 15).
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.coil_sensitivity import (
    confine_to_acs,
    estimate_smaps_calibrated,
)
from spectramr.infrastructure.physics.fft_ops import fft2c

ACS = 24
SIZE = 96


def _phantom(size: int = SIZE, n_coils: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(kspace, ground-truth maps, object support)`` for a smooth 4-coil array.

    The support comes from the ELLIPSE, never from the estimate's own RSS: the
    ground-truth maps are RSS-normalised, so thresholding their RSS selects the
    whole frame including the background, where a coil map is not defined and
    any agreement metric is noise.
    """
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
    obj = ((yy / 0.8) ** 2 + (xx / 0.7) ** 2 < 1).float()
    obj = obj * (1.0 + 0.3 * torch.sin(6 * xx) * torch.cos(5 * yy))
    centres = [(-1.3, 0.0), (1.3, 0.0), (0.0, -1.3), (0.0, 1.3)][:n_coils]
    maps = torch.stack(
        [
            torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 1.4)
            * torch.exp(1j * 1.2 * (yy * cy + xx * cx))
            for cy, cx in centres
        ]
    )[None]
    maps = maps / torch.sqrt((maps.abs() ** 2).sum(1, keepdim=True) + 1e-8)
    return fft2c(maps * obj[None, None].to(torch.complex64)), maps, obj > 0.1


def _coil_rss(maps: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((maps.abs() ** 2).sum(dim=1))


def _normalise_then_upsample(maps: torch.Tensor, size: int) -> torch.Tensor:
    """The retired order, kept here as the violation the owner must not admit."""
    maps = maps / torch.sqrt((maps.abs() ** 2).sum(dim=1, keepdim=True) + 1e-8)
    return torch.complex(
        torch.nn.functional.interpolate(
            maps.real, size=(size, size), mode="bilinear", align_corners=False
        ),
        torch.nn.functional.interpolate(
            maps.imag, size=(size, size), mode="bilinear", align_corners=False
        ),
    )


# ---------------------------------------------------------------------------
# confine_to_acs
# ---------------------------------------------------------------------------


def test_confine_to_acs_keeps_the_grid_and_zeroes_the_periphery() -> None:
    kspace, _, _ = _phantom()
    confined = confine_to_acs(kspace, ACS)

    assert confined.shape == kspace.shape, "confinement must not resize the grid"
    lo, hi = SIZE // 2 - ACS // 2, SIZE // 2 - ACS // 2 + ACS
    torch.testing.assert_close(confined[:, :, lo:hi, lo:hi], kspace[:, :, lo:hi, lo:hi])
    # Masked max, not a difference of two sums: the periphery holds ~9x more
    # bins than the ACS and float32 cancellation over them is larger than any
    # tolerance worth asserting.
    outside = confined.clone()
    outside[:, :, lo:hi, lo:hi] = 0
    assert float(outside.abs().max()) == 0.0


def test_confine_to_acs_rejects_a_non_4d_tensor() -> None:
    with pytest.raises(ValueError, match="4D kspace"):
        confine_to_acs(torch.zeros(4, SIZE, SIZE, dtype=torch.complex64), ACS)


# ---------------------------------------------------------------------------
# The order, per method
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [("power_iter", {"kernel_size": 6}), ("espirit", {"kernel_size": 6})],
)
def test_calibrated_maps_carry_unit_coil_rss(method: str, kwargs: dict) -> None:
    """RSS is 1 wherever the estimator claims support, with no lattice."""
    kspace, truth, support = _phantom()
    maps = estimate_smaps_calibrated(
        kspace, method=method, out_size=(SIZE, SIZE), acs_size=ACS, **kwargs
    )
    assert maps is not None
    # espirit zeroes out-of-support pixels by construction (step 6 of Uecker
    # et al.), so the contract is stated over the OBJECT, not over the frame.
    rss = _coil_rss(maps)[0][support]
    torch.testing.assert_close(rss, torch.ones_like(rss), atol=1e-3, rtol=0)
    assert torch.isfinite(maps.abs()).all()
    assert maps.shape == truth.shape


@pytest.mark.parametrize("method", ["power_iter", "espirit"])
def test_calibrated_maps_are_full_resolution(method: str) -> None:
    """No caller has to interpolate, because nothing comes back cropped."""
    kspace, _, _ = _phantom()
    maps = estimate_smaps_calibrated(kspace, method=method, acs_size=ACS, kernel_size=6)
    assert maps is not None
    assert maps.shape[-2:] == kspace.shape[-2:]


def test_normalise_then_upsample_violation_collapses_rss() -> None:
    """The retired order, planted: RSS must fail on a phase that rotates.

    Two unit-modulus samples 180 degrees apart interpolate through zero, so the
    midpoint of the linear interpolant has no modulus left at all. This is the
    shape the corpus hit -- measured RSS min 0.0024 against 1.0 -- and the test
    exists so reinstating the order turns something red.
    """
    node = torch.zeros(1, 2, 3, 3, dtype=torch.complex64)
    ramp = torch.tensor([0.0, torch.pi, 2 * torch.pi])
    node[:, 0] = torch.exp(1j * ramp)[None, :, None].expand(1, 3, 3)
    node[:, 1] = torch.exp(1j * ramp)[None, :, None].expand(1, 3, 3)

    broken = _normalise_then_upsample(node, SIZE)
    assert float(_coil_rss(broken).min()) < 0.1, (
        "the planted violation did not break RSS, so the assertion below proves nothing"
    )


def test_resize_renormalises_after_interpolating() -> None:
    """When ``out_size`` does fire, the divide happens last.

    Guards the same ordering on the branch a caller reaches when its k-space
    grid genuinely differs from its image grid -- the branch the three retired
    copies took unconditionally.
    """
    kspace, _, _ = _phantom()
    maps = estimate_smaps_calibrated(
        kspace, method="power_iter", out_size=(SIZE * 2, SIZE * 2), acs_size=ACS, kernel_size=6
    )
    assert maps is not None
    assert maps.shape[-2:] == (SIZE * 2, SIZE * 2)
    rss = _coil_rss(maps)
    torch.testing.assert_close(rss, torch.ones_like(rss), atol=1e-3, rtol=0)


def test_calibration_is_invariant_to_the_aliased_periphery() -> None:
    """What ACS confinement buys: the estimate cannot see the undersampling.

    The strong form of the claim, and the one worth pinning -- not "the maps are
    accurate enough" but "the maps do not depend on which peripheral bins were
    acquired". Sampling only ever sees undersampled k-space while training
    calibrates from the fully-sampled reference, so any dependence here is a
    train/sample divergence in the coil geometry itself.
    """
    kspace, truth, support = _phantom()
    mask = torch.zeros(1, 1, SIZE, SIZE)
    mask[..., ::4, :] = 1.0
    lo, hi = SIZE // 2 - ACS // 2, SIZE // 2 - ACS // 2 + ACS
    mask[..., lo:hi, :] = 1.0

    kwargs = {"method": "power_iter", "out_size": (SIZE, SIZE), "acs_size": ACS, "kernel_size": 6}
    full = estimate_smaps_calibrated(kspace, **kwargs)
    undersampled = estimate_smaps_calibrated(kspace * mask, **kwargs)
    assert full is not None and undersampled is not None
    torch.testing.assert_close(full, undersampled, atol=0, rtol=0)

    # And they track the true geometry where one is defined. Gauge-invariant:
    # a coil map is defined only up to a per-pixel phase, so the comparison is
    # |<s, s_true>| / (|s| |s_true|) rather than an elementwise difference.
    align = (undersampled[0] * truth[0].conj()).sum(0).abs() / (
        undersampled[0].abs().pow(2).sum(0).sqrt() * truth[0].abs().pow(2).sum(0).sqrt() + 1e-9
    )
    assert float(align[support].mean()) > 0.95


def test_method_none_returns_none() -> None:
    kspace, _, _ = _phantom()
    assert estimate_smaps_calibrated(kspace, method="none") is None


def test_unknown_method_raises_rather_than_degrading() -> None:
    """Non-negotiable 3: an unregistered method is an error, not a default."""
    kspace, _, _ = _phantom()
    with pytest.raises(ValueError, match="Unknown sensitivity estimation method"):
        estimate_smaps_calibrated(kspace, method="not_a_method")
