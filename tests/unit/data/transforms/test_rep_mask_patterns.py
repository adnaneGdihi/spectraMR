"""2-D repetition masks: one meaning of R, the calibration square, and the modes.

The support mimics M4Raw: 256 phase-encode columns of which 195 are acquired,
with the partial-FOV zero band split unevenly across the two edges.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch

from spectramr.data.transforms.rep_mask_patterns import (
    CARTESIAN_1D,
    R_TOLERANCE,
    SUPPORTED_PATTERNS,
    calibration_region,
    draw_rep_masks_2d,
    resolve_pattern,
)

_H, _W = 256, 256
_TWO_D = ("poisson_disk", "radial", "spiral")


def _support() -> torch.Tensor:
    support = torch.ones(_W, dtype=torch.bool)
    support[:31] = False
    support[226:] = False
    return support


def _draw(pattern: str, acceleration: float, mode: str = "independent", key: object = "a"):
    return draw_rep_masks_2d(
        pattern,
        _support(),
        _H,
        acceleration=acceleration,
        center_fraction=0.08,
        mode=mode,
        keys=(key, 0, 0),
    )


def _grid() -> torch.Tensor:
    return _support()[None, :].expand(_H, _W)


@pytest.mark.parametrize("acceleration", [4.0, 8.0, 16.0])
@pytest.mark.parametrize("pattern", _TWO_D)
def test_r_is_counted_on_the_supported_bins(pattern: str, acceleration: float) -> None:
    m1, _, achieved = _draw(pattern, acceleration)
    supported = int(_grid().sum())
    assert not (m1 & ~_grid()).any()
    assert achieved == pytest.approx(supported / int(m1.sum()))
    if pattern == "radial":  # whole spokes: within the tolerance, not to the bin
        assert abs(achieved / acceleration - 1) <= R_TOLERANCE
    else:
        assert int(m1.sum()) == round(supported / acceleration)


@pytest.mark.parametrize("pattern", _TWO_D)
def test_the_calibration_square_is_in_both_masks(pattern: str) -> None:
    calib = calibration_region(_H, _W, 0.08)
    assert int(calib.sum()) == 400  # the 20 x 20 window coil_maps_acs_size: 20 reads
    rows, cols = torch.nonzero(calib, as_tuple=True)
    assert (int(rows.min()), int(cols.min())) == (_H // 2 - 10, _W // 2 - 10)
    for mode in ("independent", "complementary", "same"):
        m1, m2, _ = _draw(pattern, 8.0, mode)
        assert m1[calib].all() and m2[calib].all(), mode


@pytest.mark.parametrize("pattern", _TWO_D)
def test_the_masks_are_genuinely_two_dimensional(pattern: str) -> None:
    m1, _, _ = _draw(pattern, 4.0)
    assert not torch.equal(m1[0], m1[_H // 4])


class TestTheModes:
    @pytest.mark.parametrize("pattern", _TWO_D)
    def test_independent_is_a_second_draw_of_the_same_size(self, pattern: str) -> None:
        m1, m2, _ = _draw(pattern, 8.0, "independent")
        assert not torch.equal(m1, m2 & _grid())
        assert abs(int((m2 & _grid()).sum()) / int(m1.sum()) - 1) <= 2 * R_TOLERANCE

    @pytest.mark.parametrize("pattern", _TWO_D)
    def test_same_is_m1(self, pattern: str) -> None:
        m1, m2, _ = _draw(pattern, 8.0, "same")
        assert torch.equal(m2 & _grid(), m1)

    def test_full_target_is_everything(self) -> None:
        _, m2, _ = _draw("poisson_disk", 8.0, "full_target")
        assert m2.all()

    @pytest.mark.parametrize("pattern", _TWO_D)
    def test_complementary_is_disjoint_outside_the_calibration_square(self, pattern: str) -> None:
        calib = calibration_region(_H, _W, 0.08)
        m1, m2, _ = _draw(pattern, 8.0, "complementary")
        assert not (m1 & m2 & ~calib).any()
        assert int((m2 & _grid()).sum()) == int(m1.sum())  # rebalanced above R=2

    def test_complementary_covers_the_support_at_r2(self) -> None:
        m1, m2, _ = _draw("poisson_disk", 2.0, "complementary")
        assert torch.equal((m1 | m2) & _grid(), _grid())

    def test_every_mode_feeds_the_same_input(self) -> None:
        firsts = [
            _draw("radial", 8.0, mode)[0] for mode in ("independent", "same", "complementary")
        ]
        assert all(torch.equal(m, firsts[0]) for m in firsts)


def test_masks_are_a_function_of_the_keys() -> None:
    assert torch.equal(_draw("radial", 8.0, key="x")[0], _draw("radial", 8.0, key="x")[0])
    assert not torch.equal(_draw("radial", 8.0, key="x")[0], _draw("radial", 8.0, key="y")[0])


def test_masks_are_identical_in_a_fresh_process() -> None:
    """CRC32 keys, not Python's per-process salted ``hash``."""
    code = (
        "import torch, hashlib\n"
        "from spectramr.data.transforms.rep_mask_patterns import draw_rep_masks_2d\n"
        "s = torch.ones(256, dtype=torch.bool); s[:31] = False; s[226:] = False\n"
        "for p in ('poisson_disk', 'radial'):\n"
        "    m1, m2, _ = draw_rep_masks_2d(p, s, 256, acceleration=8.0, center_fraction=0.08,"
        " mode='independent', keys=('a', 0, 0))\n"
        "    print(hashlib.sha256(torch.stack([m1, m2]).numpy().tobytes()).hexdigest())\n"
    )
    import hashlib
    import os

    env = {**os.environ, "PYTHONHASHSEED": "12345", "SPECTRAMR_SUPPRESS_CLINICAL_WARNING": "1"}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    ).stdout.split()
    here = []
    for pattern in ("poisson_disk", "radial"):
        m1, m2, _ = _draw(pattern, 8.0)
        here.append(hashlib.sha256(torch.stack([m1, m2]).numpy().tobytes()).hexdigest())
    assert out == here


class TestItRefuses:
    def test_a_calibration_square_that_fills_the_budget(self) -> None:
        with pytest.raises(ValueError, match="calibration region alone"):
            draw_rep_masks_2d(
                "poisson_disk",
                _support(),
                32,
                acceleration=16.0,
                center_fraction=0.08,
                mode="same",
                keys=("a", 0, 0),
            )

    def test_an_unknown_pattern(self) -> None:
        with pytest.raises(ValueError, match="Unknown sampling pattern"):
            resolve_pattern("hexagonal")

    def test_a_registry_family_it_does_not_draw(self) -> None:
        """``random_cartesian`` is points, not lines: no silent fallback to either."""
        with pytest.raises(ValueError, match="does not draw"):
            resolve_pattern("random_cartesian")


def test_names_resolve_through_the_registry() -> None:
    assert resolve_pattern("poisson") == "poisson_disk"
    assert resolve_pattern("radial_golden") == "golden_angle"
    assert resolve_pattern(CARTESIAN_1D) == CARTESIAN_1D
    assert CARTESIAN_1D in SUPPORTED_PATTERNS


def test_the_radius_memo_changes_nothing_but_time() -> None:
    from spectramr.data.transforms import rep_mask_patterns as mod

    warm = _draw("poisson_disk", 8.0)[0]
    mod._RADIUS_CACHE.clear()
    assert torch.equal(_draw("poisson_disk", 8.0)[0], warm)
    assert len(mod._RADIUS_CACHE) <= mod._RADIUS_CACHE_MAX


def test_the_calibration_region_is_the_scale_windows_one_owner() -> None:
    """``acs_window`` scales on this square; the two must not drift apart."""
    from spectramr.data.transforms.normalization import acs_window
    from spectramr.data.transforms.rep_mask_patterns import calibration_region

    region = calibration_region(256, 256, 0.08)
    rows, cols = acs_window(256, 256, 0.08)
    expected = torch.zeros(256, 256, dtype=torch.bool)
    expected[rows, cols] = True
    assert torch.equal(region, expected) and int(region.sum()) == 400
