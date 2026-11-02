"""Tests for the acquisition-keyed spatial-frequency band partition.

The load-bearing claim is that ``rho > 1`` marks content the acquisition never
measured, so a transfer gain there separates recovery from fabrication. That
claim is pinned empirically at the bottom of this file (see
``test_super_nyquist_gain_separates_recovery_from_fabrication``) rather than
asserted in prose: the partition is only useful if an exact multi-frame
inversion scores near 1 where a plausible-but-wrong signal scores near 0.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch.nn.functional import avg_pool2d, interpolate  # noqa: E402

from spectramr.infrastructure.physics.band_partition import (  # noqa: E402
    acquisition_rho,
    band_edges,
    band_masks,
    band_partition,
    band_transfer,
    super_nyquist_band_indices,
)
from spectramr.infrastructure.physics.subpixel_registration import (  # noqa: E402
    centred_fft,
    centred_freqs,
    centred_ifft,
    fourier_shift,
)
from spectramr.infrastructure.physics.virtual_fiducial import (  # noqa: E402
    VirtualFiducial,
)

# ── rho: where the acquisition Nyquist actually lands ────────────────────────


def test_rho_is_exactly_one_at_the_decimation_nyquist() -> None:
    """s-fold pooling resolves 1/(2s) cycles/pixel; that bin must read rho=1."""
    n, s = 64, 2
    rho = acquisition_rho((n, n), sr_scale=s)
    # centred layout puts DC at n // 2, so f = 1/(2s) sits n // (2*s) bins out
    assert rho[n // 2 + n // (2 * s), n // 2] == pytest.approx(1.0, abs=1e-6)
    assert rho[n // 2, n // 2] == pytest.approx(0.0, abs=1e-7)


def test_rho_reproduces_the_real_ulf_resolution_gap() -> None:
    """The grid reaches effective/grid times the acquisition Nyquist per axis.

    T1w is stored at 0.49 mm but resolved at 1.6 mm, so the 3T grid extends
    3.27x beyond anything the 64 mT scanner measured — the independently
    derived 3.3x in-plane SR factor.
    """
    rho = acquisition_rho((256, 256), voxel_mm=(0.49, 0.49), effective_voxel_mm=(1.6, 1.6))
    per_axis = float(rho[0, 128])
    assert per_axis == pytest.approx(1.6 / 0.49, rel=1e-2)


def test_rho_handles_anisotropy_per_axis() -> None:
    """A scalar cutoff would misclassify the thin axis; per-axis must not."""
    rho = acquisition_rho(
        (32, 32, 16),
        voxel_mm=(0.49, 0.49, 1.0),
        effective_voxel_mm=(1.6, 1.6, 5.0),
    )
    assert rho.shape == (32, 32, 16)
    # in-plane gap 3.27, through-plane gap 5.0 -> different reach per axis
    assert float(rho[0, 16, 8]) == pytest.approx(1.6 / 0.49, rel=1e-2)
    assert float(rho[16, 16, 0]) == pytest.approx(5.0 / 1.0, rel=1e-2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},  # neither
        {"sr_scale": 2, "voxel_mm": (1.0, 1.0), "effective_voxel_mm": (2.0, 2.0)},
        {"voxel_mm": (1.0, 1.0)},  # grid without effective
    ],
)
def test_rho_requires_exactly_one_parameterisation(kwargs: dict) -> None:
    with pytest.raises(ValueError, match=r"parameterisation|BOTH"):
        acquisition_rho((32, 32), **kwargs)


def test_rho_rejects_an_acquisition_finer_than_its_own_grid() -> None:
    """Then every band is sub-Nyquist and the probe measures nothing."""
    with pytest.raises(ValueError, match="finer than the stored grid"):
        acquisition_rho((32, 32), voxel_mm=(1.6, 1.6), effective_voxel_mm=(0.49, 0.49))


def test_rho_rejects_axis_count_mismatch() -> None:
    with pytest.raises(ValueError, match="entries to match size"):
        acquisition_rho((32, 32), voxel_mm=(1.0, 1.0, 1.0), effective_voxel_mm=(2.0, 2.0, 2.0))


# ── edges and masks ──────────────────────────────────────────────────────────


def test_nyquist_is_always_a_band_boundary() -> None:
    """No band may straddle rho=1, or it is neither measured nor unmeasured."""
    for n_sub in (1, 2, 4):
        for n_super in (1, 3):
            edges = band_edges(n_sub, n_super, rho_max=2.5)
            assert 1.0 in [pytest.approx(e) for e in edges]
            assert len(edges) == n_sub + n_super + 1
            assert list(edges) == sorted(edges)


def test_super_nyquist_indices_are_the_bands_above_one() -> None:
    edges = band_edges(2, 2, rho_max=2.0)
    assert super_nyquist_band_indices(edges) == (2, 3)


@pytest.mark.parametrize(
    "kwargs", [{"n_sub": 0}, {"n_super": 0}, {"rho_max": 1.0}, {"rho_max": 0.5}]
)
def test_band_edges_rejects_a_partition_with_no_comparison(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        band_edges(**{"n_sub": 2, "n_super": 2, "rho_max": 2.0, **kwargs})


def test_band_masks_raise_when_the_grid_cannot_reach_rho_max() -> None:
    """An underpopulated band still returns a number; that number is noise."""
    rho = acquisition_rho((32, 32), sr_scale=1)  # max rho = sqrt(2)/2*2 ~ 0.707
    with pytest.raises(ValueError, match=r"frequency bins|maximum is"):
        band_masks(rho, band_edges(2, 2, rho_max=2.0))


def test_band_masks_reject_non_monotone_edges() -> None:
    rho = acquisition_rho((32, 32), sr_scale=2)
    with pytest.raises(ValueError, match="strictly increasing"):
        band_masks(rho, (0.0, 1.0, 0.5, 2.0))


def test_bands_are_disjoint_and_cover_their_span() -> None:
    rho = acquisition_rho((64, 64), sr_scale=2)
    masks = band_masks(rho, band_edges(2, 2, rho_max=2.0))
    assert masks.shape == (4, 64, 64)
    assert masks.sum(dim=0).max() <= 1.0  # disjoint
    assert masks.sum() == (rho < 2.0).sum()  # covers [0, rho_max)


# ── partition ────────────────────────────────────────────────────────────────


def test_partition_reconstructs_the_input_when_the_bands_span_the_grid() -> None:
    x = torch.randn(2, 3, 64, 64)
    rho = acquisition_rho((64, 64), sr_scale=2)
    full = band_masks(rho, (0.0, float(rho.max()) + 1e-3))
    assert band_partition(x, full).squeeze(2) == pytest.approx(x, abs=1e-5)


def test_partition_is_differentiable() -> None:
    x = torch.randn(1, 1, 32, 32, requires_grad=True)
    masks = band_masks(acquisition_rho((32, 32), sr_scale=2), band_edges(1, 1, 1.4))
    band_partition(x, masks).pow(2).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_partition_works_in_3d() -> None:
    x = torch.randn(1, 2, 16, 16, 16)
    rho = acquisition_rho((16, 16, 16), sr_scale=2)
    masks = band_masks(rho, band_edges(1, 1, rho_max=1.4))
    assert band_partition(x, masks).shape == (1, 2, 2, 16, 16, 16)


def test_partition_rejects_masks_of_the_wrong_extent() -> None:
    masks = band_masks(acquisition_rho((32, 32), sr_scale=2), band_edges(1, 1, 1.4))
    with pytest.raises(ValueError, match="do not match the spatial extent"):
        band_partition(torch.randn(1, 1, 64, 64), masks)


# ── transfer gain ────────────────────────────────────────────────────────────


def test_transfer_is_one_against_itself_and_zero_against_noise() -> None:
    x = torch.randn(4, 1, 64, 64)
    masks = band_masks(acquisition_rho((64, 64), sr_scale=2), band_edges(2, 2, 2.0))
    assert band_transfer(x, x, masks) == pytest.approx(torch.ones(4, 4), abs=1e-5)
    indep = band_transfer(torch.randn_like(x), x, masks)
    assert indep.abs().max() < 0.25


def test_transfer_is_scale_invariant() -> None:
    """A halved-amplitude reproduction is a task-loss failure, not a transfer
    failure; keeping the two claims separable is the point of the cosine."""
    x = torch.randn(2, 1, 64, 64)
    masks = band_masks(acquisition_rho((64, 64), sr_scale=2), band_edges(2, 2, 2.0))
    assert band_transfer(0.5 * x, x, masks) == pytest.approx(band_transfer(x, x, masks), abs=1e-5)


def test_transfer_is_differentiable_in_the_prediction() -> None:
    target = torch.randn(1, 1, 32, 32)
    pred = torch.randn(1, 1, 32, 32, requires_grad=True)
    masks = band_masks(acquisition_rho((32, 32), sr_scale=2), band_edges(1, 1, 1.4))
    (1.0 - band_transfer(pred, target, masks)).mean().backward()
    assert pred.grad is not None and float(pred.grad.norm()) > 0.0


def test_transfer_honours_a_support_weight() -> None:
    x = torch.randn(1, 1, 32, 32)
    masks = band_masks(acquisition_rho((32, 32), sr_scale=2), band_edges(1, 1, 1.4))
    support = torch.zeros(1, 1, 32, 32)
    support[..., :16, :16] = 1.0
    corrupt = x.clone()
    corrupt[..., 16:, 16:] = torch.randn(16, 16)  # outside the support
    assert float(band_transfer(corrupt, x, masks, support=support).min()) > float(
        band_transfer(corrupt, x, masks).min()
    )


def test_transfer_rejects_shape_mismatch() -> None:
    masks = band_masks(acquisition_rho((32, 32), sr_scale=2), band_edges(1, 1, 1.4))
    with pytest.raises(ValueError, match="must match"):
        band_transfer(torch.randn(1, 1, 32, 32), torch.randn(1, 2, 32, 32), masks)


# ── the claim the partition exists to support ────────────────────────────────


def test_super_nyquist_gain_separates_recovery_from_fabrication() -> None:
    """Exact multi-frame inversion scores ~1 above Nyquist; a wrong signal ~0.

    The forward model is ``s``-fold pooling of ``n`` sub-pixel-shifted views.
    With n=8 and s=2 that operator is full rank on this grid, so super-Nyquist
    content is genuinely identifiable (Tsai & Huang, 1984) and an exact
    least-squares inversion must recover it. A different marker realisation is
    equally sharp and equally plausible, and must not.
    """
    h = 24
    s, n = 2, 8
    marker = (
        VirtualFiducial(im_size=(h, h), grid_spacing=8, sigma=1.0, jitter=0.35, seed=0)(1)
        .real.double()
        .contiguous()
    )
    torch.manual_seed(0)
    shifts = (torch.rand(1, n, 2) - 0.5) * 2.0

    def forward(x: torch.Tensor) -> torch.Tensor:
        shifted = fourier_shift(x.float().expand(1, n, h, h).contiguous(), shifts)
        return avg_pool2d(shifted, s).double()

    y = forward(marker)
    basis = torch.eye(h * h, dtype=torch.float64)
    op = torch.stack([forward(e.reshape(1, 1, h, h)).reshape(-1) for e in basis], dim=1)
    solved = torch.linalg.lstsq(op, y.reshape(-1, 1), driver="gelsd").solution
    solved = solved.reshape(1, 1, h, h).float()

    masks = band_masks(
        acquisition_rho((h, h), sr_scale=s), band_edges(1, 1, rho_max=1.4), min_bins=8
    )
    sup = list(super_nyquist_band_indices(band_edges(1, 1, rho_max=1.4)))
    ref = marker.float()

    def gain(pred: torch.Tensor) -> float:
        return float(band_transfer(pred, ref, masks)[:, sup].mean())

    fabricated = VirtualFiducial(im_size=(h, h), grid_spacing=8, sigma=1.0, jitter=0.35, seed=99)(
        1
    ).real
    floor = interpolate(avg_pool2d(ref, s), size=(h, h), mode="bilinear", align_corners=False)

    assert gain(solved) > 0.99, "exact inversion must recover the unmeasured bands"
    assert abs(gain(fabricated)) < 0.2, "a plausible-but-wrong signal must not"
    # Boxcar pooling is not an ideal anti-aliasing filter, so a plain
    # interpolator scores well above zero by partially unfolding the aliasing
    # a single frame retains. That floor is why an absolute gain must never be
    # quoted without it.
    assert 0.2 < gain(floor) < 0.95
    assert gain(solved) > gain(floor) + 0.2


# ── passband geometry: a Cartesian frame measures a box, not a disc ───────────


def _measured_square(h: int, s: int) -> torch.Tensor:
    f = (2 * s * centred_freqs(h, torch.device("cpu"), torch.float64)).abs()
    return torch.maximum(f.view(h, 1), f.view(1, h)) < 1.0


@pytest.mark.parametrize(("geometry", "leaks"), [("cartesian", False), ("radial", True)])
def test_super_nyquist_bands_hold_no_measured_cartesian_bin(geometry: str, leaks: bool) -> None:
    """#2413: radial bands above 1 contain the corners of the pooled frame's square."""
    h, s = 64, 2
    edges = band_edges(2, 2, 2.0)
    masks = band_masks(acquisition_rho((h, h), sr_scale=s, geometry=geometry), edges)
    sup = list(super_nyquist_band_indices(edges))
    overlap = float((masks[sup].bool() & _measured_square(h, s)).sum())
    assert (overlap > 0) is leaks


def test_zero_filled_frame_carries_no_super_nyquist_energy() -> None:
    """A frame with no content outside its measured square must score nothing there."""
    h, s = 64, 2
    m = VirtualFiducial(im_size=(h, h), grid_spacing=8, sigma=1.0, jitter=0.35, seed=0)(1).real.double()
    zf = centred_ifft(centred_fft(m) * _measured_square(h, s).to(torch.complex128)).real
    edges = band_edges(2, 2, 2.0)
    masks = band_masks(acquisition_rho((h, h), sr_scale=s, dtype=torch.float64), edges)
    sup = list(super_nyquist_band_indices(edges))
    bands = band_partition(zf, masks)[:, :, sup]
    assert float(bands.pow(2).sum()) < 1e-10 * float(band_partition(m, masks)[:, :, sup].pow(2).sum())


def test_rho_is_the_per_axis_maximum_on_a_cartesian_grid() -> None:
    rho = acquisition_rho((32, 32), voxel_mm=(0.5, 0.5), effective_voxel_mm=(1.0, 2.0))
    radial = acquisition_rho(
        (32, 32), voxel_mm=(0.5, 0.5), effective_voxel_mm=(1.0, 2.0), geometry="radial"
    )
    assert torch.all(rho <= radial + 1e-6)
    assert float(rho.max()) == pytest.approx(4.0)  # thin axis: 2 * (2.0 / 0.5) * 0.5


def test_rho_rejects_an_unknown_geometry() -> None:
    with pytest.raises(ValueError, match="geometry"):
        acquisition_rho((32, 32), sr_scale=2, geometry="spiral")  # type: ignore[arg-type]
