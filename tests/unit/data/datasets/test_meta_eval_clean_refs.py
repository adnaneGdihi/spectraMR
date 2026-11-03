"""Shape contract of the meta-evaluation ``.pt`` clean references."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from spectramr.data.datasets.meta_eval_clean_refs import load_clean_volumes


def _save(tmp_path: Path, payload: object, name: str = "ref.pt") -> Path:
    torch.save(payload, tmp_path / name)
    return tmp_path


def test_a_slice_stack_raises_and_points_to_volume_mode(tmp_path: Path) -> None:
    _save(tmp_path, torch.randn(4, 64, 64))
    with pytest.raises(ValueError, match="--mode 3d"):
        load_clean_volumes(tmp_path, max_subjects=10)


def test_a_bare_slice_gets_a_channel_axis(tmp_path: Path) -> None:
    _save(tmp_path, torch.randn(8, 8))
    [(sid, t)] = load_clean_volumes(tmp_path, max_subjects=10)
    assert sid == "ref"
    assert t.shape == (1, 8, 8)


def test_a_single_channel_slice_passes_unchanged(tmp_path: Path) -> None:
    _save(tmp_path, torch.randn(1, 8, 8))
    [(_, t)] = load_clean_volumes(tmp_path, max_subjects=10)
    assert t.shape == (1, 8, 8)


def test_explicit_coils_are_moved_to_the_channel_axis(tmp_path: Path) -> None:
    _save(tmp_path, {"image": torch.randn(8, 8, 4), "coil_axis": 2})
    [(_, t)] = load_clean_volumes(tmp_path, max_subjects=10)
    assert t.shape == (4, 8, 8)


@pytest.mark.parametrize("coil_axis", [3, -4, "0", 1.0])
def test_an_invalid_coil_axis_raises(tmp_path: Path, coil_axis: object) -> None:
    _save(tmp_path, {"image": torch.randn(4, 8, 8), "coil_axis": coil_axis})
    with pytest.raises(ValueError, match="coil_axis"):
        load_clean_volumes(tmp_path, max_subjects=10)


def test_a_complex_reference_raises_instead_of_dropping_the_phase(tmp_path: Path) -> None:
    _save(tmp_path, torch.randn(1, 8, 8, dtype=torch.complex64))
    with pytest.raises(ValueError, match="complex"):
        load_clean_volumes(tmp_path, max_subjects=10)


def test_a_volume_raises(tmp_path: Path) -> None:
    _save(tmp_path, torch.randn(1, 4, 8, 8))
    with pytest.raises(ValueError, match="--mode 3d"):
        load_clean_volumes(tmp_path, max_subjects=10)
