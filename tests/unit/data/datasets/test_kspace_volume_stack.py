"""The ``.h5`` reads and the per-slice NEX pseudo-GT volume stack (rule 7).

The volume tests write tiny synthetic M4Raw-style trees and build their keys through
``discover_m4raw_grid``, so the loader is exercised on the same ``ContentKey`` the sweep
hands it.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from spectramr.data.datasets.contrast_grid import discover_m4raw_grid
from spectramr.data.datasets.kspace_volume_stack import (
    header_voxel_mm,
    load_kspace_volume,
)


def _header(matrix=(32, 24), fov=(160.0, 180.0), spacing: float | None = 6.0) -> str:
    gap = "" if spacing is None else f"<spacingBetweenSlices>{spacing}</spacingBetweenSlices>"
    return (
        "<ismrmrdHeader><encoding><encodedSpace>"
        f"<matrixSize><x>{matrix[0]}</x><y>{matrix[1]}</y><z>1</z></matrixSize>"
        f"<fieldOfView_mm><x>{fov[0]}</x><y>{fov[1]}</y><z>5</z></fieldOfView_mm>"
        f"</encodedSpace></encoding><sequenceParameters>{gap}<TR>500</TR>"
        "</sequenceParameters></ismrmrdHeader>"
    )


def _tree(tmp_path, shape=(4, 1, 32, 24), header=None, reps=2, contrasts=("T1",)):
    rng = np.random.default_rng(0)
    base = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    for c in contrasts:
        for r in range(1, reps + 1):
            noise = 0.01 * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
            with h5py.File(tmp_path / f"2022010101_{c}{r:02d}.h5", "w") as f:
                f.create_dataset("kspace", data=(base + noise).astype(np.complex64))
                f.create_dataset("ismrmrd_header", data=np.bytes_((header or _header()).encode()))
    return discover_m4raw_grid(tmp_path, contrasts=list(contrasts), max_contrasts=len(contrasts))


def test_a_volume_round_trips_to_shape_spacing_and_contrast(tmp_path) -> None:
    grid = _tree(tmp_path, contrasts=("T1", "T2"))
    vol = load_kspace_volume(grid.keys[1])
    assert vol.contrast == "T2" and vol.content_id == "2022010101_T2"
    assert vol.magnitude.shape == (1, 1, 4, 32, 24)
    assert vol.coil_images.shape == vol.smaps.shape == (1, 1, 4, 32, 24)
    assert vol.header.voxel_mm[0] == 6.0
    assert vol.voxel_mm == (6.0, 5.0, 7.5)  # (slice, H, W): 160/32 and 180/24
    assert vol.n_reps == 2
    assert vol.provenance()["normalisation"] == "p99_over_volume"


def test_one_p99_normalises_the_whole_stack(tmp_path) -> None:
    vol = load_kspace_volume(_tree(tmp_path).keys[0])
    assert torch.quantile(vol.magnitude.flatten(), 0.99).item() == pytest.approx(1.0, rel=1e-4)
    per_slice = [torch.quantile(vol.magnitude[0, 0, s].flatten(), 0.99).item() for s in range(4)]
    assert max(per_slice) - min(per_slice) > 1e-3  # not renormalised per slice


def test_multicoil_maps_are_estimated_per_slice(tmp_path) -> None:
    vol = load_kspace_volume(_tree(tmp_path, shape=(3, 2, 32, 24)).keys[0])
    assert vol.coil_images.shape == vol.smaps.shape == (1, 2, 3, 32, 24)
    assert vol.magnitude.shape == (1, 1, 3, 32, 24)
    assert torch.isfinite(vol.magnitude).all()


def test_a_transposed_encoded_matrix_raises(tmp_path) -> None:
    """ISMRMRD fixes x (readout) to H; a y-by-x matrix is not reinterpreted."""
    with pytest.raises(ValueError, match="is not kspace H"):
        load_kspace_volume(_tree(tmp_path, header=_header((24, 32), (180.0, 160.0))).keys[0])


def test_spacing_between_slices_is_the_stamped_source(tmp_path) -> None:
    prov = load_kspace_volume(_tree(tmp_path).keys[0]).provenance()
    assert prov["spacing_source"] == "spacingBetweenSlices"
    assert prov["pe_dim"] == -1


def test_without_spacing_between_slices_the_fov_z_is_used_and_stamped(tmp_path, caplog) -> None:
    import spectramr.data.datasets.kspace_volume_stack as kvs

    kvs._warned_fov_z = False
    with caplog.at_level("WARNING"):
        vol = load_kspace_volume(_tree(tmp_path, header=_header(spacing=None)).keys[0])
    assert vol.voxel_mm == (5.0, 5.0, 7.5)
    assert vol.provenance()["spacing_source"] == "encoded_fov_z"
    assert sum("encoded FOV z" in r.getMessage() for r in caplog.records) == 1


def test_a_header_without_matrix_or_fov_raises() -> None:
    with pytest.raises(ValueError, match="lacks encodedSpace"):
        header_voxel_mm({"slice_spacing_mm": 5.0}, 32, 24, "x.h5")


def test_a_square_matrix_with_an_anisotropic_fov_follows_the_convention() -> None:
    params = {"slice_spacing_mm": 5.0, "encoded_matrix": (32, 32), "fov_mm": (200.0, 100.0, 1.0)}
    assert header_voxel_mm(params, 32, 32, "x.h5").voxel_mm == (5.0, 6.25, 3.125)


def test_a_bad_header_fails_before_any_kspace_read(tmp_path, monkeypatch) -> None:
    import spectramr.data.datasets.kspace_volume_stack as kvs

    key = _tree(tmp_path, header=_header((24, 32))).keys[0]

    def never(*_a, **_k):
        raise AssertionError("k-space was read before the header was resolved")

    monkeypatch.setattr(kvs, "load_kspace_from_h5", never)
    with pytest.raises(ValueError, match="is not kspace H"):
        load_kspace_volume(key)


def test_a_slice_content_is_refused(tmp_path) -> None:
    from dataclasses import replace

    key = replace(_tree(tmp_path).keys[0], slice_index=2)
    with pytest.raises(ValueError, match="every slice"):
        load_kspace_volume(key)


class TestSliceOnlyRepReads:
    """#616 (audit D5). Pseudo-GT averages ONE slice across every repetition.

    The target slice is derived from the slice count, so the old single pass
    loaded every rep in FULL to learn a number, held them all resident, then
    used one slice each. On an M4Raw multi-slice multi-coil acquisition that is
    the whole volume per repetition, times the number of repetitions.
    """

    @staticmethod
    def _write_rep(path, n_slices=5, n_coils=4, h=8, w=8, seed=0):
        h5py = pytest.importorskip("h5py")
        import numpy as np

        rng = np.random.default_rng(seed)
        data = rng.standard_normal((n_slices, n_coils, h, w, 2)).astype("float32")
        with h5py.File(path, "w") as f:
            f.create_dataset("kspace", data=data)
        return data

    def test_the_header_probe_reports_the_shape_without_reading(self, tmp_path):
        from spectramr.data.datasets.kspace_volume_stack import (
            kspace_shape_from_h5 as _kspace_shape_from_h5,
        )

        path = tmp_path / "rep0.h5"
        self._write_rep(path)
        assert _kspace_shape_from_h5(path) == (5, 4, 8, 8, 2)

    def test_a_slice_read_equals_the_same_slice_of_a_full_read(self, tmp_path):
        """The equivalence that makes the optimisation safe. Asserted against
        the FULL read rather than against a literal, so a change to the complex
        coercion cannot pass here while breaking one path."""
        import torch

        from spectramr.data.datasets.kspace_volume_stack import (
            load_kspace_from_h5 as _load_kspace_from_h5,
        )

        path = tmp_path / "rep0.h5"
        self._write_rep(path, n_slices=5)

        full = _load_kspace_from_h5(path)
        for idx in range(5):
            partial = _load_kspace_from_h5(path, slice_index=idx)
            assert torch.equal(partial, full[idx]), f"slice {idx} diverged"

    def test_the_partial_read_drops_the_leading_axis(self, tmp_path):
        from spectramr.data.datasets.kspace_volume_stack import (
            load_kspace_from_h5 as _load_kspace_from_h5,
        )

        path = tmp_path / "rep0.h5"
        self._write_rep(path, n_slices=5, n_coils=4)
        assert _load_kspace_from_h5(path).shape == (5, 4, 8, 8)
        assert _load_kspace_from_h5(path, slice_index=2).shape == (4, 8, 8)

    def test_a_missing_kspace_key_still_raises_keyerror_on_the_probe(self, tmp_path):
        """The per-rep error handling now spans both passes; the probe must
        raise the same class the loader did so the caller's except arms fire."""
        h5py = pytest.importorskip("h5py")

        from spectramr.data.datasets.kspace_volume_stack import (
            kspace_shape_from_h5 as _kspace_shape_from_h5,
        )

        path = tmp_path / "empty.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("something_else", data=[1, 2, 3])
        with pytest.raises(KeyError):
            _kspace_shape_from_h5(path)

    def test_a_missing_file_raises_filenotfounderror_on_the_probe(self, tmp_path):
        from spectramr.data.datasets.kspace_volume_stack import (
            kspace_shape_from_h5 as _kspace_shape_from_h5,
        )

        with pytest.raises(FileNotFoundError):
            _kspace_shape_from_h5(tmp_path / "nope.h5")
