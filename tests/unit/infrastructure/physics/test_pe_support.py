"""``pe_support``: which phase-encode columns a coil-noise fit may use.

The per-scan fit (``calibrate_scan``) and the per-batch sigma
(``sigma_from_kspace_batch``) both read their support here, so the band test,
the ACS run and the fraction each has one owner. The accuracy of the fits that
consume these lives in ``test_scan_coil_calibration.py``.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.pe_support import (
    NonBandSupportError,
    acs_band,
    batch_fit_columns,
    fit_support,
    map_leak_bound,
    pe_band_fraction,
    pe_support_of,
)


def _mask(width: int, sampled: list[int] | range) -> torch.Tensor:
    mask = torch.zeros(width, dtype=torch.bool)
    mask[list(sampled)] = True
    return mask


def _random_with_acs(width: int = 64, acs: int = 8, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    mask = torch.rand(width, generator=g) < 0.3
    c = width // 2
    mask[c - acs // 2 : c + acs // 2] = True
    mask[c - acs // 2 - 1] = False
    mask[c + acs // 2] = False
    return mask


class TestTheBand:
    def test_a_band_is_its_own_fraction(self) -> None:
        assert pe_band_fraction(_mask(10, range(2, 9))) == pytest.approx(0.7)

    def test_a_gap_raises_and_names_the_opt_in(self) -> None:
        with pytest.raises(NonBandSupportError, match="support_policy='acs'"):
            pe_band_fraction(_mask(8, [0, 2, 3]))

    def test_the_support_is_read_off_the_data(self) -> None:
        kspace = torch.ones(2, 3, 4, 6, dtype=torch.complex64)
        kspace[..., 4:] = 0
        assert pe_support_of(kspace).tolist() == [True] * 4 + [False] * 2


class TestTheAcsRun:
    def test_it_is_the_run_through_the_centre(self) -> None:
        mask = _random_with_acs(acs=8)
        assert acs_band(mask) == (28, 35)

    def test_an_unsampled_centre_raises(self) -> None:
        with pytest.raises(NonBandSupportError, match="centre"):
            acs_band(_mask(16, [0, 1, 2]))

    def test_band_policy_refuses_a_non_band_support(self) -> None:
        with pytest.raises(NonBandSupportError, match="gaps"):
            fit_support(_random_with_acs(), "band")

    def test_acs_policy_restricts_a_non_band_support(self) -> None:
        columns, fraction, restricted = fit_support(_random_with_acs(acs=8), "acs")
        assert restricted and fraction == pytest.approx(8 / 64)
        assert columns.nonzero().flatten().tolist() == list(range(28, 36))

    def test_acs_policy_fits_a_band_whole(self) -> None:
        band = _mask(64, range(0, 48))
        columns, fraction, restricted = fit_support(band, "acs")
        assert not restricted and fraction == pytest.approx(0.75) and torch.equal(columns, band)

    def test_a_run_narrower_than_the_calibration_window_raises(self) -> None:
        with pytest.raises(NonBandSupportError, match="narrower"):
            fit_support(_random_with_acs(acs=8), "acs", min_width=24)

    def test_an_unknown_policy_raises(self) -> None:
        with pytest.raises(ValueError, match="support_policy"):
            fit_support(_mask(4, [0, 1]), "whole")


class TestTheBatchForm:
    """Per sample, on the device, agreeing with the 1-D owner above."""

    def _batch(self, masks: list[torch.Tensor]) -> torch.Tensor:
        return (
            torch.ones(len(masks), 2, 5, masks[0].numel(), dtype=torch.complex64)
            * torch.stack(masks)[:, None, None, :]
        )

    def test_it_agrees_with_fit_support_per_sample(self) -> None:
        masks = [_random_with_acs(seed=s) for s in range(3)] + [_mask(64, range(8, 56))]
        columns, fraction = batch_fit_columns(self._batch(masks), "acs")
        for i, mask in enumerate(masks):
            ref_columns, ref_fraction, _ = fit_support(mask, "acs")
            assert torch.equal(columns[i], ref_columns)
            assert float(fraction[i]) == pytest.approx(ref_fraction)

    def test_band_policy_refuses_a_gap_in_any_sample(self) -> None:
        masks = [_mask(64, range(8, 56)), _random_with_acs()]
        with pytest.raises(NonBandSupportError):
            batch_fit_columns(self._batch(masks), "band")

    def test_acs_policy_refuses_a_sample_without_its_centre(self) -> None:
        masks = [_mask(64, [0, 5, 9])]
        with pytest.raises(NonBandSupportError, match="centre"):
            batch_fit_columns(self._batch(masks), "acs")


class TestTheLeakBound:
    def test_constant_maps_have_no_leak(self) -> None:
        maps = torch.ones(1, 4, 8, 8, dtype=torch.complex64) / 2
        assert map_leak_bound(maps, 0.5) == pytest.approx(0.0, abs=1e-6)

    def test_a_global_phase_ramp_is_not_a_turn(self) -> None:
        """A unit coil vector is defined up to a phase; only its direction counts."""
        ramp = torch.exp(1j * torch.linspace(0, 3, 8))[None, None, None, :]
        maps = (torch.ones(1, 4, 8, 8) / 2).to(torch.complex64) * ramp
        assert map_leak_bound(maps, 0.5) == pytest.approx(0.0, abs=1e-5)

    def test_it_scales_with_the_kernel_width(self) -> None:
        angle = torch.linspace(0, 1, 16)
        maps = torch.zeros(1, 2, 4, 16, dtype=torch.complex64)
        maps[:, 0] = torch.cos(angle)
        maps[:, 1] = torch.sin(angle)
        step = map_leak_bound(maps, 1.0)
        assert step == pytest.approx(1 / 15, rel=1e-3)
        assert map_leak_bound(maps, 0.1) == pytest.approx(10 * step)


def test_a_mask_along_the_rows_is_refused_by_name() -> None:
    """A row mask leaves every column with energy, so the column fit would skip the
    sampled-fraction correction; the fit refuses it and names the knob."""
    from spectramr.infrastructure.physics.pe_support import NonBandSupportError, batch_fit_columns

    kspace = torch.randn(1, 2, 16, 16, dtype=torch.complex64)
    kspace[:, :, ::2, :] = 0  # every other ROW missing, every column present
    with pytest.raises(NonBandSupportError, match="mask_direction"):
        batch_fit_columns(kspace, "acs")
    columns = torch.randn(1, 2, 16, 16, dtype=torch.complex64)
    columns[:, :, :, 2:6] = 0  # a column gap: the acs policy handles it
    batch_fit_columns(columns, "acs")


def test_a_two_dimensional_mask_is_refused_by_name() -> None:
    """A Poisson-disc or pseudo-radial mask leaves every column with some energy, so
    the column fit read a fully sampled band and skipped the 1/f correction; it
    returned a sigma instead of raising before the guard existed."""
    from spectramr.infrastructure.physics.pe_support import NonBandSupportError, batch_fit_columns

    gen = torch.Generator().manual_seed(0)
    kspace = torch.randn(1, 2, 16, 16, dtype=torch.complex64, generator=gen)
    holes = torch.rand(16, 16, generator=gen) < 0.5
    holes[6:10, 6:10] = False  # a centred calibration square
    kspace[:, :, holes] = 0
    for policy in ("band", "acs"):
        with pytest.raises(NonBandSupportError, match="2-D mask"):
            batch_fit_columns(kspace, policy)


def test_rows_zero_everywhere_are_not_holes() -> None:
    """Rows absent from every column (padding) are not a 2-D mask; a column gap still fits."""
    from spectramr.infrastructure.physics.pe_support import batch_fit_columns

    kspace = torch.randn(1, 2, 16, 16, dtype=torch.complex64)
    kspace[:, :, :, 2:6] = 0
    kspace[:, :, 0, :] = 0
    _, fraction = batch_fit_columns(kspace, "acs")
    assert float(fraction[0]) == pytest.approx(10 / 16)


def _line_mask(width: int, sampled: list[int] | range, height: int = 16) -> torch.Tensor:
    """A whole-line ``(1, 1, 1, W)`` bool sampling mask."""
    mask = torch.zeros(1, 1, 1, width, dtype=torch.bool)
    mask[..., list(sampled)] = True
    return mask


def _measure(mask: torch.Tensor, height: int = 16) -> torch.Tensor:
    gen = torch.Generator().manual_seed(1)
    data = torch.randn(1, 2, height, mask.shape[-1], dtype=torch.complex64, generator=gen)
    return data * mask


def test_an_exact_zero_inside_an_acquired_line_is_a_hole_to_a_data_read_but_not_to_the_mask() -> (
    None
):
    """The robust_ssdu_selfcal crash at iteration 905: a whole-line measurement with one
    sample that is zero in every coil. Read off the data it looks like a 2-D mask."""
    from spectramr.infrastructure.physics.pe_support import NonBandSupportError, batch_fit_columns

    mask = _line_mask(16, [0, 1, 2, 3, 6, 7, 8, 9, 12, 13])
    kspace = _measure(mask)
    kspace[:, :, 5, 8] = 0  # an acquired column, one row, every coil
    with pytest.raises(NonBandSupportError, match="2-D mask"):
        batch_fit_columns(kspace, "acs")
    columns, fraction = batch_fit_columns(kspace, "acs", acquired=mask)
    reference, reference_fraction = batch_fit_columns(_measure(mask), "acs")
    assert torch.equal(columns, reference)
    assert torch.equal(fraction, reference_fraction)


@pytest.mark.parametrize("dtype", [torch.bool, torch.float32, torch.complex64])
def test_the_acquired_mask_may_be_any_dtype(dtype) -> None:
    from spectramr.infrastructure.physics.pe_support import batch_fit_columns

    mask = _line_mask(16, range(4, 12))
    columns, fraction = batch_fit_columns(_measure(mask), "acs", acquired=mask.to(dtype))
    assert float(fraction[0]) == pytest.approx(8 / 16)
    assert int(columns.sum()) == 8


def test_a_two_dimensional_acquired_mask_is_still_refused() -> None:
    from spectramr.infrastructure.physics.pe_support import NonBandSupportError, batch_fit_columns

    gen = torch.Generator().manual_seed(0)
    mask = torch.rand(1, 1, 16, 16, generator=gen) < 0.5
    mask[..., 6:10, 6:10] = True
    kspace = torch.randn(1, 2, 16, 16, dtype=torch.complex64, generator=gen) * mask
    for policy in ("band", "acs"):
        with pytest.raises(NonBandSupportError, match="2-D mask"):
            batch_fit_columns(kspace, policy, acquired=mask)


def test_an_acquired_mask_running_along_the_rows_is_still_refused_by_name() -> None:
    from spectramr.infrastructure.physics.pe_support import NonBandSupportError, batch_fit_columns

    mask = torch.zeros(1, 1, 16, 16, dtype=torch.bool)
    mask[..., ::2, :] = True  # every other ROW acquired, every column present
    kspace = torch.randn(1, 2, 16, 16, dtype=torch.complex64) * mask
    with pytest.raises(NonBandSupportError, match="mask_direction"):
        batch_fit_columns(kspace, "acs", acquired=mask)
