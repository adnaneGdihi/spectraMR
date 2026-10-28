"""Regression test for F10/E35 (smoke_audit_20260521 round 5) —
``TorchIOQueueBuilder`` must drop subjects whose spatial extent is
smaller than the configured ``patch_size``.

Root cause: ``stage2_ldm_ulf_to_hf`` (and similar 3D experiments)
crashed with ``5D target_batch has a zero-sized dim: shape=(1, 1,
128, 128, 0)`` because ``tio.UniformSampler`` silently produced
empty patches when a manifest volume had fewer Z-slices than
``patch_size[2]``. The strategy's error message suggested manually
filtering the manifest; F10 implements that filter generically at
the queue-build layer.

Fix: ``TorchIOQueueBuilder._filter_patch_compatible_subjects(dataset,
patch_size)`` walks every subject, inspects its image data shape,
drops those with any spatial extent < corresponding patch size.
Returns the original dataset unchanged when nothing was dropped
(preserves SubjectsDataset transform behavior).

Tests:
- 2-D patch_size keeps all subjects (no z constraint).
- 3-D patch_size drops subjects with insufficient z.
- Empty-result case still returns a valid dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torchio as tio

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from spectramr.data.builders.torchio_queue_builder import TorchIOQueueBuilder


def _make_subject(shape: tuple[int, int, int, int]) -> tio.Subject:
    """Build a torchio subject with a single image of the given shape."""
    return tio.Subject(image=tio.ScalarImage(tensor=torch.randn(*shape)))


def test_filter_keeps_subjects_meeting_patch_size() -> None:
    """Subjects with extent ≥ patch_size on every spatial axis must be kept."""
    subjects = [
        _make_subject((1, 256, 256, 18)),  # ample extent
        _make_subject((1, 256, 256, 32)),
        _make_subject((1, 256, 256, 18)),
    ]
    ds = tio.SubjectsDataset(subjects)
    out = TorchIOQueueBuilder._filter_patch_compatible_subjects(
        ds, patch_size=(128, 128, 16)
    )
    # No filtering needed → returns the original dataset.
    assert len(out) == 3


def test_filter_drops_short_z_volumes() -> None:
    """The 2026-05-20 stage2_ldm crash was a (..., 128, 128, 0) patch
    on a volume with insufficient z. Pinned: subjects with depth
    smaller than patch_size[2] must be dropped."""
    subjects = [
        _make_subject((1, 256, 256, 18)),  # ok
        _make_subject((1, 256, 256, 2)),   # depth=2 < patch_depth=16 → drop
        _make_subject((1, 256, 256, 32)),  # ok
    ]
    ds = tio.SubjectsDataset(subjects)
    out = TorchIOQueueBuilder._filter_patch_compatible_subjects(
        ds, patch_size=(128, 128, 16)
    )
    assert len(out) == 2, (
        f"Subjects with depth < patch_depth must be dropped; got "
        f"{len(out)} (expected 2 — the 2 ok ones)"
    )


def test_filter_drops_short_h_w_volumes() -> None:
    """Spatial extent guard must apply to H and W too, not just depth."""
    subjects = [
        _make_subject((1, 256, 256, 18)),  # ok
        _make_subject((1, 64, 256, 18)),   # H=64 < patch_h=128 → drop
        _make_subject((1, 256, 64, 18)),   # W=64 < patch_w=128 → drop
    ]
    ds = tio.SubjectsDataset(subjects)
    out = TorchIOQueueBuilder._filter_patch_compatible_subjects(
        ds, patch_size=(128, 128, 16)
    )
    assert len(out) == 1


def test_filter_returns_dataset_when_nothing_dropped() -> None:
    """When all subjects pass, return the original dataset object —
    avoids needlessly rebuilding the transform chain. This is a
    no-op-fast-path guarantee that downstream code (queue, transforms)
    sees the same dataset identity."""
    subjects = [_make_subject((1, 256, 256, 18))]
    ds = tio.SubjectsDataset(subjects)
    out = TorchIOQueueBuilder._filter_patch_compatible_subjects(
        ds, patch_size=(128, 128, 16)
    )
    assert out is ds, (
        "When no filtering is needed the original dataset must be "
        "returned unchanged (identity), to preserve transform chain."
    )


def test_filter_handles_2d_patch_size() -> None:
    """2-D patch_size (no z constraint) only checks H/W. Pin to make
    sure the dim-count handling doesn't accidentally check too many axes."""
    subjects = [
        _make_subject((1, 256, 256, 1)),  # depth=1 but 2D patch — ok
        _make_subject((1, 64, 64, 1)),    # H=W=64 < patch=128 — drop
    ]
    ds = tio.SubjectsDataset(subjects)
    out = TorchIOQueueBuilder._filter_patch_compatible_subjects(
        ds, patch_size=(128, 128)  # 2-D
    )
    assert len(out) == 1


class _LazyShapeDataset:
    """Mimics ``UniversalMRIDataset``: exposes ``.index`` with per-record
    ``shape`` and a ``__getitem__`` that would LOAD a volume (here: raises).

    Used to pin the F-OOM 2026-06-08 fix — the patch-compatibility filter must
    read extents from the manifest ``shape`` and NEVER materialise voxel data
    when shapes are present (the old probe held all 1024 loaded subjects in a
    list → ~39 GB → host OOM at queue-build, independent of num_workers)."""

    def __init__(self, shapes):
        self.index = [{"shape": list(s), "file_id": f"v{i}"} for i, s in enumerate(shapes)]
        self.loads = 0

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):  # pragma: no cover - must NOT be called
        self.loads += 1
        raise AssertionError(
            "patch-compat filter loaded voxel data despite manifest shapes "
            "being available (F-OOM regression)"
        )


def test_fast_path_filters_from_index_shapes_without_loading():
    """The M4Raw case: 1024 volumes, 2-D patch — all compatible. Must return the
    dataset unchanged WITHOUT loading a single volume."""
    ds = _LazyShapeDataset([(18, 4, 256, 256)] * 1024)
    out = TorchIOQueueBuilder._filter_patch_compatible_subjects(ds, patch_size=(256, 256, 1))
    assert out is ds
    assert ds.loads == 0, "filter must not materialise voxel data when shapes are present"
    assert len(out.index) == 1024


def test_fast_path_drops_short_hw_from_shapes():
    """H/W extent guard still applies — checked from manifest shape, no load."""
    ds = _LazyShapeDataset([(18, 4, 256, 256), (18, 4, 64, 256), (18, 4, 256, 64)])
    out = TorchIOQueueBuilder._filter_patch_compatible_subjects(ds, patch_size=(128, 128, 1))
    assert ds.loads == 0
    assert len(out.index) == 1  # only the 256x256 volume fits a 128x128 patch


def test_fast_path_drops_short_depth_for_3d():
    """Depth (slice) guard applies for 3-D patches, from the leading shape dim."""
    ds = _LazyShapeDataset([(18, 4, 256, 256), (2, 4, 256, 256)])
    out = TorchIOQueueBuilder._filter_patch_compatible_subjects(ds, patch_size=(128, 128, 16))
    assert ds.loads == 0
    assert len(out.index) == 1  # depth-2 volume dropped (2 < 16)


def test_filter_reads_nifti_manifest_shape_as_chwd() -> None:
    """NIfTI records store [C, H, W, D]: a 3-D patch must compare against D, not C or W."""
    ds = _LazyShapeDataset([(1, 128, 128, 56), (1, 128, 128, 8)])
    for r in ds.index:
        r["primary_path"] = f"{r['file_id']}.nii.gz"
    out = TorchIOQueueBuilder._filter_patch_compatible_subjects(ds, (64, 64, 16))
    assert [r["file_id"] for r in out.index] == ["v0"]
    assert ds.loads == 0


def test_require_a_subject_fits_names_the_patch_when_every_subject_is_dropped() -> None:
    from spectramr.data.builders.torchio_queue_builder import _require_a_subject_fits

    with pytest.raises(ValueError, match=r"All 2 validation subject\(s\).*patch_size=\(64, 64, 64\)"):
        _require_a_subject_fits("validation", 2, [], (64, 64, 64))


def test_require_a_subject_fits_is_silent_when_a_subject_survives() -> None:
    from spectramr.data.builders.torchio_queue_builder import _require_a_subject_fits

    ds = tio.SubjectsDataset([_make_subject((1, 256, 256, 18))])
    _require_a_subject_fits("train", 2, ds, (64, 64, 16))
