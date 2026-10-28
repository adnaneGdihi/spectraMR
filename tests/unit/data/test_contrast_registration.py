"""Unit tests for spectramr.data.contrast_registration.

A synthetic subject carries NEX averages (``..00.h5``): a T1 whose anatomy is
displaced by a planted ``(slices, rows, cols)`` shift against its T2, and a FLAIR.
The fit must recover the shift, the written T1 must re-measure as aligned under the
independent phase-correlation check, and the file must keep the source's layout,
header and attributes. Acquired repetitions of either contrast are refused.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from spectramr.data.contrast_registration import (
    RigidFit,
    acs_disc,
    apply_fit,
    group_registrable,
    register_split,
    voxel_spacing,
)
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from tests.fixtures import m4raw_phantom as ph

pytest.importorskip("SimpleITK")
REPO = Path(__file__).resolve().parents[3]
PLANT = (1.0, 2.6, -1.4)


def _check_module():
    path = REPO / "scripts/data/contrast_registration_check.py"
    spec = importlib.util.spec_from_file_location("contrast_registration_check", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _subject(src: Path, *, real_storage: bool = False) -> Path:
    src.mkdir(parents=True, exist_ok=True)
    ph.write(src / "2022000001_T200.h5", ph.phantom(1.0))
    ph.write(src / "2022000001_FLAIR00.h5", ph.phantom(0.7))
    ph.write(
        src / "2022000001_T100.h5",
        ph.phantom(-1.0, PLANT),
        real_storage=real_storage,
        patient_id="p1",
    )
    return src


def _register(src: Path, dst: Path, **kw) -> list[dict]:
    opts = {"reference": "T1", "target": "T2", "acs_fraction": 0.02, "max_rotation_deg": 2.0}
    return register_split(src, dst, **{**opts, **kw})


def test_a_planted_shift_is_recovered_from_the_acs_disc(tmp_path: Path) -> None:
    (record,) = _register(_subject(tmp_path / "src"), tmp_path / "dst")
    assert record["status"] == "registered"
    assert np.allclose(record["shift_px"], [-p for p in PLANT], atol=0.25)
    assert record["slice_offset"] == -1
    assert record["rotation_deg"] < 0.5


def test_the_written_reference_re_measures_as_aligned(tmp_path: Path) -> None:
    src = _subject(tmp_path / "src")
    _register(src, tmp_path / "dst")
    check = _check_module()
    before = check.measure(src, "m4raw", 10)["pairs"][("T1", "T2")]["slices"]
    after = check.measure(tmp_path / "dst", "m4raw", 10)["pairs"][("T1", "T2")]["slices"]
    assert min(shift for shift, _ in before) >= 2  # 2.6 px, read to the whole pixel
    assert after == [(0, 0)] * 5


def test_the_output_keeps_the_source_layout_and_records_the_fit(tmp_path: Path) -> None:
    src = _subject(tmp_path / "src")
    _register(src, tmp_path / "dst")
    out = tmp_path / "dst" / "2022000001_T100.h5"
    assert not out.is_symlink()
    with h5py.File(src / "2022000001_T100.h5") as a, h5py.File(out) as b:
        assert b["kspace"].dtype == a["kspace"].dtype
        assert b["kspace"].shape == a["kspace"].shape
        assert b["ismrmrd_header"][()] == a["ismrmrd_header"][()]
        assert b.attrs["patient_id"] == "p1"
        assert b.attrs["registration_reference_to_target"] == "T1->T2"
        assert b.attrs["registration_target_file"] == "2022000001_T200.h5"
        assert not b.attrs["registration_rotation_applied"]
        valid = b.attrs["registration_valid_slices"]
        assert valid[:-1].all() and not valid[-1]
        assert not np.abs(b["kspace"][-1]).any()


def test_real_imag_storage_is_written_back_in_kind(tmp_path: Path) -> None:
    src = _subject(tmp_path / "src", real_storage=True)
    (record,) = _register(src, tmp_path / "dst")
    with h5py.File(tmp_path / "dst" / "2022000001_T100.h5") as f:
        assert f["kspace"].dtype == np.float32 and f["kspace"].shape[-1] == 2
    assert np.allclose(record["shift_px"], [-p for p in PLANT], atol=0.25)


def test_every_other_file_is_a_symlink_to_the_original(tmp_path: Path) -> None:
    src = _subject(tmp_path / "src")
    _register(src, tmp_path / "dst")
    for name in ("2022000001_T200.h5", "2022000001_FLAIR00.h5"):
        link = tmp_path / "dst" / name
        assert link.is_symlink() and link.resolve() == (src / name).resolve()


def test_a_rotation_over_the_limit_is_refused_and_not_written(tmp_path: Path) -> None:
    (record,) = _register(_subject(tmp_path / "src"), tmp_path / "dst", max_rotation_deg=0.0)
    assert record["status"].startswith("refused")
    assert not (tmp_path / "dst" / "2022000001_T100.h5").exists()


def test_a_reference_on_a_different_grid_is_refused(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    ph.write(src / "2022000001_T200.h5", ph.phantom(1.0))
    ph.write(src / "2022000001_T100.h5", ph.phantom(-1.0)[:-2])
    (record,) = _register(src, tmp_path / "dst")
    assert record["status"].startswith("refused: grid")
    assert not (tmp_path / "dst" / "2022000001_T100.h5").exists()


def test_a_reference_without_a_target_is_linked_unregistered(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    ph.write(src / "2022000002_T100.h5", ph.phantom(-1.0))
    (record,) = _register(src, tmp_path / "dst")
    assert record["status"].startswith("no T2 file")
    assert (tmp_path / "dst" / "2022000002_T100.h5").is_symlink()


def test_a_non_empty_destination_is_refused(tmp_path: Path) -> None:
    src = _subject(tmp_path / "src")
    (tmp_path / "dst").mkdir()
    (tmp_path / "dst" / "stale.h5").touch()
    with pytest.raises(FileExistsError, match="not empty"):
        _register(src, tmp_path / "dst")


def test_apply_fit_moves_content_toward_increasing_index() -> None:
    kspace = fft2c(torch.randn(6, 2, 16, 16, dtype=torch.complex64))
    shifted, valid = apply_fit(kspace, RigidFit((1.0, 2.0, -3.0), 1, 0.0))
    expected = torch.roll(ifft2c(kspace), shifts=(2, -3), dims=(-2, -1))
    assert torch.allclose(ifft2c(shifted)[1:], expected[:-1], atol=1e-5)
    assert not valid[0] and valid[1:].all()


def test_acs_disc_keeps_the_declared_fraction_about_the_centre() -> None:
    disc = acs_disc(256, 256, 0.02)
    assert disc[128, 128] and not disc[0, 0]
    assert disc.float().mean().item() == pytest.approx(0.02, rel=0.05)


def test_voxel_spacing_reads_the_prefixed_header_and_refuses_a_matrix_mismatch() -> None:
    assert voxel_spacing(ph.header(), ph.SIZE, ph.SIZE) == (5.0, 0.9375, 0.9375)
    assert voxel_spacing(ph.header(prefix=""), ph.SIZE, ph.SIZE) == (5.0, 0.9375, 0.9375)
    with pytest.raises(ValueError, match="does not match kspace"):
        voxel_spacing(ph.header(), ph.SIZE, ph.SIZE * 2)


@pytest.mark.parametrize("name", ["2022000001_T101.h5", "2022000001_T202.h5"])
def test_an_acquired_repetition_of_either_contrast_stops_the_split(
    tmp_path: Path, name: str
) -> None:
    src = _subject(tmp_path / "src")
    ph.write(src / name, ph.phantom(1.0))
    with pytest.raises(ValueError, match="acquired repetition; register the NEX averages"):
        _register(src, tmp_path / "dst")
    assert not (tmp_path / "dst").exists()


def test_a_repetition_of_an_unregistered_contrast_is_only_linked(tmp_path: Path) -> None:
    src = _subject(tmp_path / "src")
    ph.write(src / "2022000001_FLAIR01.h5", ph.phantom(0.7))
    (record,) = _register(src, tmp_path / "dst")
    assert record["status"] == "registered"
    assert (tmp_path / "dst" / "2022000001_FLAIR01.h5").is_symlink()


def test_images_reconstructed_from_the_old_kspace_are_dropped(tmp_path: Path) -> None:
    src = _subject(tmp_path / "src")
    with h5py.File(src / "2022000001_T100.h5", "a") as f:
        f["reconstruction_rss"] = np.ones((ph.SLICES, ph.SIZE, ph.SIZE), np.float32)
    _register(src, tmp_path / "dst")
    with h5py.File(tmp_path / "dst" / "2022000001_T100.h5") as f:
        assert "reconstruction_rss" not in f
        assert list(f.attrs["dropped_kspace_derived_datasets"]) == ["reconstruction_rss"]


def test_grouping_reads_names_and_refuses_repetitions_without_writing(tmp_path: Path) -> None:
    src = _subject(tmp_path / "src")
    groups = group_registrable(src, reference="T1", target="T2")
    assert {c: [p.name for p in ps] for c, ps in groups["2022000001"].items()} == {
        "FLAIR": ["2022000001_FLAIR00.h5"],
        "T1": ["2022000001_T100.h5"],
        "T2": ["2022000001_T200.h5"],
    }
    ph.write(src / "2022000001_T101.h5", ph.phantom(-1.0))
    with pytest.raises(ValueError, match="acquired repetition"):
        group_registrable(src, reference="T1", target="T2")
    assert not (tmp_path / "dst").exists()
