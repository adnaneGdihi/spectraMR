"""The slice route serves what the group route serves, slice for slice (#1757, #2673).

The n2n and self_supervised M4Raw cohorts moved to ``data.slice_level_records``.
For the same (group, slice) the two routes must hand the model the same input,
target, second repetition, coil maps and masks, and validation must still write
one row per slice under the same identity keys. What legitimately differs is
pinned beside it: the r2r draw (a global-RNG call of a different shape) and the
k-space scale (``test_m4raw_slice_route_scan_level.py``).
"""

from __future__ import annotations

import pytest
import torch

from spectramr.data.collation.strategies import ImageCollateStrategy
from spectramr.data.datasets.m4raw_dataset import M4RawRepetitionDataset, _pe_support
from spectramr.data.transforms.complementary_rep_masking import ComplementaryRepMasking
from spectramr.infrastructure.reporting.per_slice import SLICE_KEY
from spectramr.infrastructure.training.validation_rows import expand_depth, identifier_strings
from tests.fixtures import m4raw_slice_volumes as vols

_PATIENTS = ("2022061001", "2022061002")


def _files(tmp_path, *, maps: bool) -> tuple[list, dict]:
    files = []
    for g, patient in enumerate(_PATIENTS):
        files += vols.write_group(tmp_path / "data", patient, seed=g + 1)
    extra: dict = {}
    if maps:
        vols.write_maps(tmp_path / "maps", files)
        extra = {"coil_maps_root": tmp_path / "maps", "coil_maps_missing": "raise"}
    return files, extra


def _pair(tmp_path, *, maps: bool = True, transform=None, **kw):
    files, extra = _files(tmp_path, maps=maps)
    kw = {"single_contrast": True, "coil_processing_mode": "none", **extra, **kw}
    return (
        M4RawRepetitionDataset(files, transform=transform, **kw),
        M4RawRepetitionDataset(files, transform=transform, slice_level_records=True, **kw),
    )


_LOO = {
    "target_mode": "phase_aligned_mean",
    "nex_target_exclude_input": True,
    "emit_alternate_input": True,
}
_ROUTES = {
    "loo_mean_alt_maps": (_LOO, None),
    "loo_mean_masked": (
        _LOO,
        ComplementaryRepMasking(acceleration=2.0, center_fraction=0.08, mode="independent"),
    ),
    "rep_pair": ({"target_mode": "rep_pair", "nex_target_exclude_input": True}, None),
    "nullspace_committed": ({"target_mode": "nullspace_r2r", "r2r_whitening": "committed"}, None),
    "nullspace_selfcal": (
        {"target_mode": "nullspace_r2r", "r2r_whitening": "self_calibrated"},
        None,
    ),
}


def _fixed_null_direction(s_tilde: torch.Tensor, generator=None) -> torch.Tensor:
    """``draw_null_direction`` with one fixed vector instead of a per-slice draw.

    The real draw is one Gaussian vector per slice from the global RNG, so a
    six-slice call and a one-slice call consume it differently; pinning the
    vector leaves only the deterministic part of the split to compare.
    """
    g = torch.arange(1, s_tilde.shape[1] + 1, dtype=torch.float64)
    g = torch.complex(g, g.flip(0)).to(s_tilde.dtype).reshape(1, -1, 1, 1)
    v = g - s_tilde * (s_tilde.conj() * g).sum(1, keepdim=True)
    return v / v.abs().pow(2).sum(1, keepdim=True).sqrt()


@pytest.mark.parametrize("route", list(_ROUTES))
def test_every_image_of_a_slice_record_is_that_slice_of_the_group(
    tmp_path, monkeypatch, route
) -> None:
    """Equal up to float32 round-off: the phase-alignment sums run over a batch
    of six planes on one route and one plane on the other (measured <= 2e-5 on
    k-space values of order 10^2)."""
    from spectramr.infrastructure.physics import coil_split_r2r

    monkeypatch.setattr(coil_split_r2r, "draw_null_direction", _fixed_null_direction)
    kw, transform = _ROUTES[route]
    vol, sl = _pair(tmp_path, transform=transform, **kw)
    volumes = [vol[g] for g in range(len(vol))]
    assert len(sl) == len(vol) * vols.SLICES
    for k, record in enumerate(sl.index):
        g, s = record["group_index"], record["slice_index"]
        subject, whole = sl[k], volumes[g]
        names = set(subject.get_images_names())
        assert names == set(whole.get_images_names()), (route, names)
        assert {"input", "target", "sensitivity_complex"} <= names
        for name in names:
            expected = whole[name].data[..., s : s + 1]
            torch.testing.assert_close(
                subject[name].data, expected, rtol=1e-5, atol=1e-4, msg=f"{route}:{name}"
            )
        for key in ("subject_id", "contrast", "repetition_group", "file_id"):
            assert subject[key] == whole[key], key
        assert int(subject["slice_index"]) == s


def test_the_comparison_sees_a_planted_off_by_one_slice(tmp_path) -> None:
    """The planted shape: a record serving its neighbour's slice. The tolerance
    above must not absorb it, or the equivalence test would pass vacuously."""
    vol, sl = _pair(tmp_path, maps=False, **_LOO)
    whole = vol[0]
    for name in ("input", "target", "input_alt"):
        neighbour = whole[name].data[..., 3:4]
        assert not torch.allclose(sl[2][name].data, neighbour, rtol=1e-5, atol=1e-4), name


def test_the_r2r_draw_differs_but_its_signal_is_the_same_slice(tmp_path) -> None:
    """r2r adds ``alpha z`` and subtracts ``z / alpha`` with ``z`` from the global
    RNG, so the halves are not equal across routes (a draw of another shape);
    at ``alpha = 1`` their mean is the acquired repetition, which must match."""
    vol, sl = _pair(tmp_path, maps=False, target_mode="r2r", r2r_alpha=1.0)
    whole = vol[0]
    for s in range(vols.SLICES):
        subject = sl[s]
        mean = (subject["input"].data + subject["target"].data) / 2
        torch.testing.assert_close(
            mean, ((whole["input"].data + whole["target"].data) / 2)[..., s : s + 1]
        )
    assert not torch.equal(sl[0]["input"].data, vol[0]["input"].data[..., 0:1])


def test_acquired_line_support_is_per_slice_on_a_record() -> None:
    """``_pe_support`` is the union over slices on a volume and one slice's on a
    record. M4Raw's sampled columns always carry noise, so the two agree on real
    data; a column that is exactly zero on one slice only is where they part."""
    kspace = torch.from_numpy(vols.volume(seed=3))
    assert torch.equal(_pe_support(kspace), _pe_support(kspace[2:3]))
    kspace[2, :, :, 5] = 0
    assert bool(_pe_support(kspace)[5]) and not bool(_pe_support(kspace[2:3])[5])


def _rows(dataset) -> list[tuple[str, ...]]:
    """What ``validation_batches.csv`` keys on: the real collate, ``expand_depth``
    and the row writer's identifier rendering, at validation batch size 1."""
    collate = ImageCollateStrategy(squeeze_depth_dim=True).collate
    batches = (collate([dataset[i]]) for i in range(len(dataset)))
    rows = []
    for batch in expand_depth(batches):
        cells = identifier_strings(batch, 1)
        rows.append(tuple(cells[key] for key in SLICE_KEY))
    return rows


def test_validation_writes_the_same_slice_rows_on_both_routes(tmp_path) -> None:
    vol, sl = _pair(tmp_path, maps=False, **_LOO)
    on_groups, on_slices = _rows(vol), _rows(sl)
    assert len(on_slices) == len(_PATIENTS) * vols.SLICES
    assert len(set(on_slices)) == len(on_slices), "a slice was scored twice"
    assert sorted(on_slices) == sorted(on_groups)
    assert all(all(cell != "" for cell in row) for row in on_slices)
    assert {row[2] for row in on_slices} == {str(s) for s in range(vols.SLICES)}
