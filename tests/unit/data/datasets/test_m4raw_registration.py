"""Translation registration of M4Raw repetitions (#2849)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from spectramr.data.datasets.m4raw_dataset import M4RawRepetitionDataset, _average_reps
from spectramr.data.datasets.m4raw_registration import (
    REGISTRATION_SLICES,
    estimate_shift,
    registration_window,
    shift_kspace,
)
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

_ZERO_COLUMNS = 6


def _phantom_kspace(slices: int = 4, coils: int = 3, size: int = 32) -> torch.Tensor:
    """``(S, C, H, W)`` k-space of a blob and a box, partial phase FOV zero-filled."""
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    image = torch.exp(-((yy - 14.0) ** 2 / 30 + (xx - 17.0) ** 2 / 50))
    image = image + 0.5 * ((yy - 9).abs() < 3) * ((xx - 22).abs() < 4)
    vols = []
    for s in range(slices):
        coil_images = [
            image * (1 + 0.2 * s) * (1 + 0.3 * c) * np.exp(1j * 0.9 * c) for c in range(coils)
        ]
        vols.append(torch.stack(coil_images).to(torch.complex64))
    kspace = fft2c(torch.stack(vols))
    kspace[..., :_ZERO_COLUMNS] = 0
    return kspace


def _rss(kspace: torch.Tensor) -> torch.Tensor:
    return ifft2c(kspace).abs().pow(2).sum(dim=-3).sqrt()


class TestEstimateShift:
    @pytest.mark.parametrize("shift", [(0.3, -0.7), (1.14, 0.2), (-2.6, 1.5), (7.3, -12.4)])
    def test_recovers_a_known_sub_pixel_shift(self, shift) -> None:
        kspace = _phantom_kspace()
        dy, dx = estimate_shift(kspace, shift_kspace(kspace, *shift))
        assert dy == pytest.approx(shift[0], abs=0.05)
        assert dx == pytest.approx(shift[1], abs=0.05)

    def test_is_blind_to_each_coils_receiver_phase_and_to_noise(self) -> None:
        kspace = _phantom_kspace()
        gen = torch.Generator().manual_seed(0)
        moved = shift_kspace(kspace, 0.6, -0.4)
        moved = moved * torch.exp(1j * 6 * torch.rand(1, 3, 1, 1, generator=gen))
        noise = torch.randn(moved.shape, generator=gen) + 1j * torch.randn(
            moved.shape, generator=gen
        )
        moved = moved + 0.3 * moved.abs().mean() * noise * (kspace != 0)
        dy, dx = estimate_shift(kspace, moved.to(torch.complex64))
        assert (dy, dx) == pytest.approx((0.6, -0.4), abs=0.05)

    def test_unmoved_repetitions_read_zero(self) -> None:
        kspace = _phantom_kspace()
        assert estimate_shift(kspace, kspace) == pytest.approx((0.0, 0.0), abs=1e-9)

    def test_single_coil_volumes_are_accepted(self) -> None:
        kspace = _phantom_kspace(coils=1)[:, 0]
        assert estimate_shift(kspace, shift_kspace(kspace, 1.0, 0.5)) == pytest.approx(
            (1.0, 0.5), abs=0.05
        )

    def test_mismatched_grids_raise(self) -> None:
        kspace = _phantom_kspace()
        with pytest.raises(ValueError, match="one shape"):
            estimate_shift(kspace, kspace[:2])


class TestShiftKspace:
    def test_never_acquired_columns_stay_exactly_zero(self) -> None:
        moved = shift_kspace(_phantom_kspace(), 0.37, -1.21)
        assert torch.equal(moved[..., :_ZERO_COLUMNS], torch.zeros_like(moved[..., :_ZERO_COLUMNS]))
        assert bool((moved[..., _ZERO_COLUMNS:] != 0).all())

    def test_an_integer_shift_is_a_circular_image_shift(self) -> None:
        kspace = _phantom_kspace()
        kspace[..., :_ZERO_COLUMNS] = fft2c(ifft2c(kspace))[..., :_ZERO_COLUMNS]
        moved = ifft2c(shift_kspace(kspace, 2.0, -3.0))
        torch.testing.assert_close(
            moved, torch.roll(ifft2c(kspace), shifts=(2, -3), dims=(-2, -1)), atol=1e-4, rtol=0
        )

    def test_shifting_back_restores_the_input(self) -> None:
        kspace = _phantom_kspace()
        back = shift_kspace(shift_kspace(kspace, 0.4, 1.3), -0.4, -1.3)
        torch.testing.assert_close(back, kspace, atol=1e-4, rtol=0)


def test_registration_window_is_central_and_clipped() -> None:
    assert registration_window(18) == range(7, 7 + REGISTRATION_SLICES)
    assert registration_window(2) == range(0, 2)


class TestAverageRepsRegistered:
    """``_average_reps(..., shifts=...)`` translates every repetition onto the anchor."""

    def _moved_reps(self):
        kspace = _phantom_kspace()
        shifts = [(0.0, 0.0), (0.8, -0.5), (-1.1, 0.4)]
        reps = [shift_kspace(kspace, dy, dx) for dy, dx in shifts]
        reps = [r * np.exp(1j * phase) for r, phase in zip(reps, (0.0, 1.7, -2.3), strict=True)]
        return kspace, shifts, reps

    def test_registered_mean_removes_the_ghost(self) -> None:
        kspace, shifts, reps = self._moved_reps()
        ghosted = _average_reps(reps, "phase_aligned_mean")
        registered = _average_reps(reps, "phase_aligned_mean", shifts=shifts)
        truth = _rss(kspace)
        ghost = float((_rss(ghosted) - truth).norm() / truth.norm())
        assert float((_rss(registered) - truth).norm() / truth.norm()) < 1e-3 < 0.02 < ghost

    def test_rep_pair_target_is_registered_onto_the_input(self) -> None:
        kspace, shifts, reps = self._moved_reps()
        pair = _average_reps(reps, "rep_pair", exclude_index=0, shifts=shifts)
        torch.testing.assert_close(_rss(pair), _rss(kspace), atol=1e-4, rtol=1e-4)

    def test_anchor_is_the_excluded_input(self) -> None:
        _, shifts, reps = self._moved_reps()
        loo = _average_reps(reps, "phase_aligned_mean", exclude_index=1, shifts=shifts)
        torch.testing.assert_close(_rss(loo), _rss(reps[1]), atol=1e-4, rtol=1e-4)

    def test_a_shift_count_that_does_not_match_raises(self) -> None:
        _, shifts, reps = self._moved_reps()
        with pytest.raises(ValueError, match="registration shifts"):
            _average_reps(reps, "phase_aligned_mean", shifts=shifts[:2])


def _write_moved_group(tmp_path, shifts, *, stem="2022_T1"):
    kspace = _phantom_kspace(slices=6)
    files, reps = [], []
    for i, (dy, dx) in enumerate(shifts, start=1):
        rep = shift_kspace(kspace, dy, dx).numpy()
        path = tmp_path / f"{stem}{i:02d}.h5"
        with h5py.File(str(path), "w") as f:
            f.create_dataset("kspace", data=rep)
        files.append(path)
        reps.append(torch.from_numpy(rep))
    return kspace, files, reps


def _dataset(files, **kwargs):
    return M4RawRepetitionDataset(
        h5_files=files,
        single_contrast=True,
        coil_processing_mode="none",
        use_repetitions=True,
        target_mode="phase_aligned_mean",
        **kwargs,
    )


def _served_target(subject) -> torch.Tensor:
    data = subject["target"].data  # (2C, H, W, S) real interleaved
    complex_ = torch.complex(data[0::2], data[1::2]).permute(3, 0, 1, 2)
    return complex_


class TestDatasetRegistration:
    SHIFTS = ((0.0, 0.0), (0.9, -0.6), (-0.7, 1.2))

    def test_target_is_built_from_registered_repetitions(self, tmp_path) -> None:
        kspace, files, reps = _write_moved_group(tmp_path, self.SHIFTS)
        target = _served_target(_dataset(files, nex_registration="translation")[0])
        truth = _rss(kspace)
        assert float((_rss(target) - truth).norm() / truth.norm()) < 1e-3
        # Planted: the same group unregistered is the ghosted target #2849 measured.
        ghosted = _served_target(_dataset(files)[0])
        assert float((_rss(ghosted) - truth).norm() / truth.norm()) > 0.02
        torch.testing.assert_close(ghosted, _average_reps(reps, "phase_aligned_mean"))

    def test_slice_route_uses_the_volume_routes_shifts(self, tmp_path) -> None:
        _, files, _ = _write_moved_group(tmp_path, self.SHIFTS)
        volume = _dataset(files, nex_registration="translation")
        volume[0]
        slices = _dataset(files, nex_registration="translation", slice_level_records=True)
        slices[0]
        slices[len(slices) - 1]
        (volume_shifts,) = volume._shift_cache.values()
        (slice_shifts,) = slices._shift_cache.values()
        assert slice_shifts == volume_shifts
        assert volume_shifts[1] == pytest.approx(self.SHIFTS[1], abs=0.05)

    def test_unknown_registration_raises(self, tmp_path) -> None:
        _, files, _ = _write_moved_group(tmp_path, self.SHIFTS)
        with pytest.raises(ValueError, match="Unknown nex_registration"):
            _dataset(files, nex_registration="affine")

    def test_split_modes_refuse_registration(self, tmp_path) -> None:
        _, files, _ = _write_moved_group(tmp_path, self.SHIFTS)
        with pytest.raises(ValueError, match="nothing to register"):
            M4RawRepetitionDataset(
                h5_files=files,
                single_contrast=True,
                coil_processing_mode="none",
                target_mode="r2r",
                nex_registration="translation",
            )


def test_group_shifts_reads_zero_for_the_anchor_and_matches_estimate_shift() -> None:
    from spectramr.data.datasets.m4raw_registration import group_shifts

    anchor = _phantom_kspace()
    moved = shift_kspace(anchor, 1.5, -0.5)
    assert group_shifts([anchor, moved]) == [(0.0, 0.0), estimate_shift(anchor, moved)]
