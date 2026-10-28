"""``CoilMapSource``: the recipe a dataset declares is the recipe it is served.

The file-or-estimate behaviour under the default recipe is pinned in
``test_m4raw_coil_maps.py``; this module pins the recipe leg: a file of the
other recipe raises under either missing-policy, and the ``estimate`` fallback
runs the declared recipe, so a run before and after the store exists agrees.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.data.datasets.m4raw_coil_map_source import CoilMapSource
from spectramr.data.datasets.m4raw_coil_maps import (
    CoilMapMismatchError,
    estimate_group_maps,
    save_coil_maps,
)
from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.infrastructure.physics.scan_coil_calibration import DEFAULT_CALIBRATION

_C, _H, _W = 4, 32, 44  # non-square


def _reps(seed: int = 0) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, _H), torch.linspace(-1, 1, _W), indexing="ij")
    centres = [(-0.6, -0.6), (0.6, -0.6), (-0.6, 0.6), (0.6, 0.6)]
    sens = torch.stack([torch.exp(-((xx - a) ** 2 + (yy - b) ** 2)) for a, b in centres])
    anatomy = ((xx**2 + yy**2) < 0.5).float() * 20.0
    out = []
    for _ in range(3):
        noise = torch.randn(1, _C, _H, _W, dtype=torch.complex64, generator=gen)
        image = (sens * anatomy).to(torch.complex64)[None] + noise
        out.append(fft2c(image))
    return out


def _calibration() -> dict:
    return {**DEFAULT_CALIBRATION, "acs_size": 20}


def test_the_recipe_is_validated_at_construction(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown coil-map recipe"):
        CoilMapSource(tmp_path, "raise", _calibration(), "median")


@pytest.mark.parametrize("missing", ["raise", "estimate"])
def test_a_file_of_the_other_recipe_raises(tmp_path, missing) -> None:
    """Planted: an all-repetitions store read by an input_repetition arm."""
    maps, meta = estimate_group_maps(_reps(), _calibration())
    save_coil_maps(tmp_path / "2022061001_T1.pt", maps, meta)
    source = CoilMapSource(tmp_path, missing, _calibration(), "input_repetition")
    with pytest.raises(CoilMapMismatchError, match="recipe 'all_reps_mean'"):
        source.maps_for("2022061001_T101", _reps())


def test_the_estimate_fallback_runs_the_declared_recipe(tmp_path) -> None:
    reps = _reps()
    source = CoilMapSource(tmp_path, "estimate", _calibration(), "input_repetition")
    got, tag = source.maps_for("2022061001_T101", reps)
    expected, _ = estimate_group_maps(reps, _calibration(), "input_repetition")
    assert tag == "estimated"
    torch.testing.assert_close(got, expected)


def test_a_missing_file_under_raise_fails_and_under_estimate_does_not(tmp_path) -> None:
    """The n2n / self-supervised arms declare ``coil_maps_missing: raise`` (#2661): an absent
    store must stop the run, never fall back to a loader-side estimate that changes the data."""
    reps = _reps()
    with pytest.raises(FileNotFoundError, match="coil_maps_missing is 'raise'"):
        CoilMapSource(tmp_path, "raise", _calibration()).maps_for("2022061001_T101", reps)

    maps, source = CoilMapSource(tmp_path, "estimate", _calibration()).maps_for(
        "2022061001_T101", reps
    )
    assert source == "estimated" and maps.shape[-2:] == (_H, _W)


def test_refuse_absent_maps_names_every_missing_group_under_raise(tmp_path) -> None:
    source = CoilMapSource(tmp_path, "raise", _calibration())
    with pytest.raises(FileNotFoundError, match=r"2 coil-map file\(s\) missing") as exc:
        source.refuse_absent_maps(["2022061001_T101", "2022061002_T101", "2022061001_T102"])
    assert "2022061001_T1.pt" in str(exc.value) and "2022061002_T1.pt" in str(exc.value)


def test_refuse_absent_maps_passes_when_every_file_exists(tmp_path) -> None:
    maps, meta = estimate_group_maps(_reps(), _calibration())
    save_coil_maps(tmp_path / "2022061001_T1.pt", maps, meta)
    CoilMapSource(tmp_path, "raise", _calibration()).refuse_absent_maps(["2022061001_T101"])


def test_refuse_absent_maps_never_refuses_under_estimate(tmp_path) -> None:
    CoilMapSource(tmp_path, "estimate", _calibration()).refuse_absent_maps(["2022061001_T101"])
