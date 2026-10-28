"""Unit tests for SliceDataset.

Tests the pure-logic portions that do not require real .npy data on disk:
  - extract_contrast_from_filename: canonical contrast extraction.
  - CONTRAST_TO_IDX: completeness and collision-free.
  - SliceDataset construction validation.
  - __len__ with file_list.
  - __getitem__ key/shape contract using tiny synthetic temp .npy files.
  - Unknown contrast falls back to 'unknown'.

Real-dataset tests (scanning a directory) are marked skip because the
cluster data path is not available in CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torchio as tio

from spectramr.data.datasets.slice_dataset import (
    CONTRAST_TO_IDX,
    KNOWN_CONTRASTS,
    extract_contrast_from_filename,
)

# ── extract_contrast_from_filename ───────────────────────────────────────────


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("sub-0015_FLAIR_slice084.npy", "FLAIR"),
        ("sub-001_T1w_slice01.npy", "T1w"),
        ("sub-001_T2w_slice10.npy", "T2w"),
        ("sub-001_DWI_slice05.npy", "DWI"),
        ("sub-001_PD_slice03.npy", "PD"),
        ("sub-001_T1_slice01.npy", "T1w"),  # alias → T1w
        ("sub-001_T2_slice01.npy", "T2w"),  # alias → T2w
        ("sub-001_nocontrast_slice01.npy", "unknown"),
    ],
    ids=["FLAIR", "T1w", "T2w", "DWI", "PD", "T1-alias", "T2-alias", "unknown"],
)
def test_extract_contrast_from_filename(filename: str, expected: str) -> None:
    assert extract_contrast_from_filename(filename) == expected


def test_known_contrasts_set_is_non_empty() -> None:
    assert len(KNOWN_CONTRASTS) > 0


# ── CONTRAST_TO_IDX ───────────────────────────────────────────────────────────


def test_contrast_to_idx_contains_unknown() -> None:
    assert "unknown" in CONTRAST_TO_IDX


def test_contrast_to_idx_values_are_unique() -> None:
    values = list(CONTRAST_TO_IDX.values())
    assert len(values) == len(set(values)), "Duplicate indices in CONTRAST_TO_IDX"


# ── SliceDataset construction ─────────────────────────────────────────────────


def _make_npy_files(tmp_path: Path, n: int = 3, h: int = 16, w: int = 16) -> list[str]:
    files = []
    for i in range(n):
        arr = np.random.rand(2, h, w).astype(np.float32)
        p = tmp_path / f"sub-{i:04d}_T2w_slice{i:03d}.npy"
        np.save(str(p), arr)
        files.append(str(p))
    return files


def test_canary_construction_with_file_list(tmp_path: Path) -> None:
    files = _make_npy_files(tmp_path, n=3)
    from spectramr.data.datasets.slice_dataset import SliceDataset

    ds = SliceDataset(file_list=files, normalize=False)
    assert len(ds) == 3


def test_construction_raises_on_empty_file_list() -> None:
    from spectramr.data.datasets.slice_dataset import SliceDataset

    with pytest.raises(RuntimeError, match="No .npy files found"):
        SliceDataset(file_list=[])


def test_construction_requires_data_dir_or_file_list() -> None:
    from spectramr.data.datasets.slice_dataset import SliceDataset

    with pytest.raises((ValueError, RuntimeError)):
        SliceDataset()  # neither data_dir nor file_list


# ── __getitem__ key/shape contract ────────────────────────────────────────────


def test_getitem_returns_torchio_subject(tmp_path: Path) -> None:
    files = _make_npy_files(tmp_path, n=2)
    from spectramr.data.datasets.slice_dataset import SliceDataset

    ds = SliceDataset(file_list=files, normalize=False)
    item = ds[0]
    assert isinstance(item, tio.Subject)


def test_getitem_has_input_and_target_keys(tmp_path: Path) -> None:
    files = _make_npy_files(tmp_path, n=2)
    from spectramr.data.datasets.slice_dataset import SliceDataset

    ds = SliceDataset(file_list=files, normalize=False)
    item = ds[0]
    assert "input" in item
    assert "target" in item


def test_getitem_transform_failure_raises_not_silently_skipped(
    tmp_path: Path,
) -> None:
    """A failing transform must surface, never return the untransformed subject.

    Silently returning untransformed data trains on unnormalized / unaugmented
    input while looking healthy -- CLAUDE.md pitfall #16 (facade). Dataloader
    hardening regression (2026-07-17).
    """
    files = _make_npy_files(tmp_path, n=1)
    from spectramr.data.datasets.slice_dataset import SliceDataset

    def _boom(_subject: object) -> object:
        raise ValueError("transform boom")

    ds = SliceDataset(file_list=files, normalize=False, transform=_boom)
    with pytest.raises(RuntimeError, match=r"[Tt]ransform"):
        ds[0]


def test_getitem_input_output_shapes(tmp_path: Path) -> None:
    """Each slice is (2, H, W); after split input=[1,H,W,1], target=[1,H,W,1]."""
    files = _make_npy_files(tmp_path, n=2, h=16, w=16)
    from spectramr.data.datasets.slice_dataset import SliceDataset

    ds = SliceDataset(file_list=files, normalize=False)
    item = ds[0]
    assert item["input"].data.shape == (1, 16, 16, 1)
    assert item["target"].data.shape == (1, 16, 16, 1)


def test_getitem_extracts_contrast(tmp_path: Path) -> None:
    files = _make_npy_files(tmp_path, n=2)
    from spectramr.data.datasets.slice_dataset import SliceDataset

    ds = SliceDataset(file_list=files, normalize=False)
    item = ds[0]
    assert item["contrast"] == "T2w"


def test_getitem_bad_npy_shape_raises(tmp_path: Path) -> None:
    """Shape [3, H, W] is invalid (expected [2, H, W])."""
    from spectramr.data.datasets.slice_dataset import SliceDataset

    bad = tmp_path / "sub-0001_T1w_slice001.npy"
    np.save(str(bad), np.random.rand(3, 8, 8).astype(np.float32))
    ds = SliceDataset(file_list=[str(bad)], normalize=False)
    with pytest.raises(ValueError, match="Expected shape"):
        ds[0]


# ── Real dataset skip ─────────────────────────────────────────────────────────


@pytest.mark.skip(reason="Requires real .npy dataset files on disk (cluster only)")
def test_real_dataset_contrast_distribution() -> None:  # pragma: no cover
    pass


# ── SliceVolumeDataset (3D-volume → per-2D-slice wrapper) ──────────────────────
#
# Pins the stage-1 LDM VAE fix (2026-06-22): a 2D AutoencoderKL fed a whole
# [C,H,W,D] volume crashes in conv2d. The wrapper expands each volume into D
# depth-1 [C,H,W,1] slice samples so batch_size=1 = one slice (no OOM).

from spectramr.data.datasets.slice_dataset import SliceVolumeDataset  # noqa: E402


class _MockVolumeBase:
    """Minimal volumetric base: input/target [C=1,H,W,D]; slice k has value k
    (input) / k+100 (target) so sliced content is verifiable."""

    def __init__(
        self, depths: list[int], with_shape: bool = False, hw: int = 4
    ) -> None:
        self.index: list[dict] = []
        self._vols: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.load_count = 0
        for i, d in enumerate(depths):
            rec: dict = {"primary_path": f"/no/such/vol{i}.nii", "file_id": f"vol{i}"}
            if with_shape:
                rec["shape"] = [1, hw, hw, d]
            self.index.append(rec)
            base = torch.arange(d).float().view(1, 1, 1, d).expand(1, hw, hw, d).clone()
            self._vols.append((base, base + 100.0))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> tio.Subject:
        self.load_count += 1
        vin, vtg = self._vols[i]
        return tio.Subject(
            input=tio.ScalarImage(tensor=vin),
            target=tio.ScalarImage(tensor=vtg),
            file_id=self.index[i]["file_id"],
            path=self.index[i]["primary_path"],
        )


def test_slice_volume_expands_volume_to_slices() -> None:
    base = _MockVolumeBase([5], with_shape=True)
    ds = SliceVolumeDataset(base)
    assert len(ds) == 5  # one depth-5 volume → 5 slices


def test_slice_volume_total_across_volumes() -> None:
    base = _MockVolumeBase([3, 4, 2], with_shape=True)
    ds = SliceVolumeDataset(base)
    assert len(ds) == 9  # 3 + 4 + 2


def test_slice_volume_item_is_depth_one_and_correct_slice() -> None:
    base = _MockVolumeBase([6], with_shape=True, hw=4)
    ds = SliceVolumeDataset(base)
    item = ds[3]  # 4th slice of the only volume
    inp = item["input"].tensor
    tgt = item["target"].tensor
    assert inp.shape == (1, 4, 4, 1)  # depth squeezed to a singleton
    assert torch.allclose(inp, torch.full_like(inp, 3.0))  # slice index 3
    assert torch.allclose(tgt, torch.full_like(tgt, 103.0))  # target = +100
    assert int(item["slice_index"]) == 3
    assert item["file_id"] == "vol0"  # metadata carried through


def test_slice_volume_depth_from_shape_avoids_load_at_init() -> None:
    base = _MockVolumeBase([4, 4], with_shape=True)
    SliceVolumeDataset(base)
    assert base.load_count == 0  # shape metadata → no volume materialised at init


def test_slice_volume_depth_fallback_loads_when_no_shape() -> None:
    # no 'shape' + the path doesn't exist → header read fails → loads the volume
    base = _MockVolumeBase([3], with_shape=False)
    ds = SliceVolumeDataset(base)
    assert len(ds) == 3
    assert base.load_count >= 1


def test_slice_volume_requires_base_index() -> None:
    class _NoIndex:
        def __len__(self) -> int:
            return 0

        def __getitem__(self, i):  # pragma: no cover
            raise NotImplementedError

    with pytest.raises(AttributeError, match="expose `.index`"):
        SliceVolumeDataset(_NoIndex())


# ---------------------------------------------------------------------------
# dry_iter() — TorchIO Queue compatibility (regression: slat_slab crash,
# 2026-06-22 "'SliceVolumeDataset' object has no attribute 'dry_iter'")
# ---------------------------------------------------------------------------


def test_dry_iter_returns_one_stub_subject_per_window() -> None:
    base = _MockVolumeBase([4, 6], with_shape=True)  # 4 + 6 = 10 windows
    ds = SliceVolumeDataset(base)
    subjects = ds.dry_iter()
    assert isinstance(subjects, list)
    assert len(subjects) == len(ds) == 10
    assert all(isinstance(s, tio.Subject) for s in subjects)
    # built from metadata only — no voxel load (Queue calls this to size epochs)
    assert base.load_count == 0


# ---------------------------------------------------------------------------
# Axis canonicalisation — regression: sub-0011 T1w stored [C, D, H, W]
# (slice axis permuted into H) crashed the stage-1 VAE (2026-06-22).
# ---------------------------------------------------------------------------


class _TransposedVolumeBase:
    """A volume stored depth-FIRST ``[C, D, H, W]`` with a square in-plane
    (``H == W``) and a different slice count ``D`` — the sub-0011 permutation.
    The manifest ``shape`` records the (transposed) on-disk order."""

    def __init__(self, slice_count: int, inplane: int) -> None:
        self.load_count = 0
        self._shape = [1, slice_count, inplane, inplane]
        self.index = [
            {
                "primary_path": "/no/such/sub-0011.nii",
                "file_id": "sub-0011",
                "shape": self._shape,
            }
        ]
        d = slice_count
        # value == slice index along the (mis-placed) depth axis 1
        self._vol = (
            torch.arange(d)
            .float()
            .view(1, d, 1, 1)
            .expand(1, d, inplane, inplane)
            .clone()
        )

    def __len__(self) -> int:
        return 1

    def __getitem__(self, i: int) -> tio.Subject:
        self.load_count += 1
        return tio.Subject(
            input=tio.ScalarImage(tensor=self._vol),
            file_id="sub-0011",
            path="/no/such/sub-0011.nii",
        )


def test_canonicalises_transposed_volume_to_depth_last() -> None:
    # sub-0011: stored [C, D=5, H=8, W=8]. It must be read as a depth-5 volume
    # of 8x8 slices, NOT a depth-8 volume of 5x8 slices.
    base = _TransposedVolumeBase(slice_count=5, inplane=8)
    ds = SliceVolumeDataset(base)
    assert len(ds) == 5  # depth 5 (the odd axis), not 8 from the stale shape[-1]
    item = ds[0]
    t = item["input"].tensor  # [C, H, W, 1] after canonicalisation + window
    assert t.shape[1:3] == (8, 8)  # in-plane 8x8, not the permuted 5x8


def test_canonicalisation_is_identity_for_wellformed_volume() -> None:
    # A normal [C, H=4, W=4, D=6] volume (depth already last) is untouched —
    # the fix must not disturb the 31 correctly-oriented cohort volumes.
    base = _MockVolumeBase([6], with_shape=True, hw=4)
    ds = SliceVolumeDataset(base)
    assert len(ds) == 6
    item = ds[3]
    assert item["input"].tensor.shape == (1, 4, 4, 1)
    assert torch.allclose(item["input"].tensor, torch.full((1, 4, 4, 1), 3.0))


def test_slice_volume_lru_cache_bounded() -> None:
    base = _MockVolumeBase([2, 2, 2], with_shape=True)
    ds = SliceVolumeDataset(base, cache_size=1)
    for g in range(len(ds)):
        ds[g]
    assert len(ds._cache) <= 1  # LRU never exceeds the bound


# ── slab_depth > 1 (thin-slab windows for 3D slab models) ─────────────────────


def test_slice_volume_slab_depth_windows_non_overlapping() -> None:
    base = _MockVolumeBase([6], with_shape=True)
    ds = SliceVolumeDataset(base, slab_depth=3)
    assert len(ds) == 2  # depth 6 / slab 3 = 2 non-overlapping windows


def test_slice_volume_slab_window_shape_and_content() -> None:
    base = _MockVolumeBase([6], with_shape=True, hw=4)
    ds = SliceVolumeDataset(base, slab_depth=3)
    first = ds[0]["input"].tensor
    second = ds[1]["input"].tensor
    assert first.shape == (1, 4, 4, 3)  # depth kept at K=3
    # window 0 = slices [0,1,2], window 1 = slices [3,4,5]
    assert torch.allclose(first[0, 0, 0, :], torch.tensor([0.0, 1.0, 2.0]))
    assert torch.allclose(second[0, 0, 0, :], torch.tensor([3.0, 4.0, 5.0]))
    assert int(ds[1]["slice_index"]) == 3  # window start


def test_slice_volume_slab_drops_trailing_partial() -> None:
    base = _MockVolumeBase([7], with_shape=True)
    ds = SliceVolumeDataset(base, slab_depth=3)
    assert len(ds) == 2  # 7 // 3 = 2 windows; the 7th slice is dropped


def test_slice_volume_slab_depth_exceeds_volume_raises() -> None:
    base = _MockVolumeBase([2], with_shape=True)
    with pytest.raises(ValueError, match="cannot form a window"):
        SliceVolumeDataset(base, slab_depth=3)


def test_slice_volume_invalid_slab_depth_raises() -> None:
    base = _MockVolumeBase([4], with_shape=True)
    with pytest.raises(ValueError, match="slab_depth must be >= 1"):
        SliceVolumeDataset(base, slab_depth=0)


# ── WS2: cache_size lever (no numerics change; fewer whole-volume re-decodes) ──


def test_slice_cache_holds_all_volumes_no_redecodes() -> None:
    """With a cache large enough to hold every volume, an interleaved
    (shuffle-like) window access decodes each volume exactly once."""
    base = _MockVolumeBase([2, 2, 2], with_shape=True)
    ds = SliceVolumeDataset(base, cache_size=3)
    # index_map order: 0,1→vol0  2,3→vol1  4,5→vol2. Interleave the volumes.
    for i in [0, 2, 4, 1, 3, 5]:
        _ = ds[i]
    assert base.load_count == 3  # one decode per volume, no thrash


def test_slice_cache_two_thrashes_under_interleave() -> None:
    """The default cache_size=2 cannot hold 3 interleaved volumes, so the
    evicted volume is re-decoded — the throughput cliff slice_cache_size fixes."""
    base = _MockVolumeBase([2, 2, 2], with_shape=True)
    ds = SliceVolumeDataset(base, cache_size=2)
    for i in [0, 2, 4, 1, 3, 5]:
        _ = ds[i]
    assert base.load_count > 3


class _StampedVolumeBase(_MockVolumeBase):
    """Each load returns a new stamp, standing in for a fresh augmentation draw."""

    def __getitem__(self, i: int) -> tio.Subject:
        subject = super().__getitem__(i)
        subject["draw"] = self.load_count
        return subject


def test_slice_volume_redraw_reloads_after_one_epoch_of_windows() -> None:
    base = _StampedVolumeBase([4], with_shape=True)
    ds = SliceVolumeDataset(base, cache_size=8, redraw=True)
    first_epoch = [ds[i]["draw"] for i in range(4)]
    second_epoch = [ds[i]["draw"] for i in range(4)]
    assert first_epoch == [1, 1, 1, 1]
    assert second_epoch == [2, 2, 2, 2]


def test_slice_volume_without_redraw_keeps_the_resident_volume() -> None:
    base = _StampedVolumeBase([4], with_shape=True)
    ds = SliceVolumeDataset(base, cache_size=8)
    assert {ds[i % 4]["draw"] for i in range(12)} == {1}
    assert base.load_count == 1


def test_slice_volume_redraw_budget_is_split_across_workers(monkeypatch) -> None:
    from types import SimpleNamespace

    import spectramr.data.datasets.slice_dataset as mod

    monkeypatch.setattr(mod, "get_worker_info", lambda: SimpleNamespace(num_workers=2))
    base = _StampedVolumeBase([4], with_shape=True)
    ds = SliceVolumeDataset(base, cache_size=8, redraw=True)
    # A worker sees about half of the windows per epoch, so it redraws after 2.
    assert [ds[i]["draw"] for i in range(4)] == [1, 1, 2, 2]
