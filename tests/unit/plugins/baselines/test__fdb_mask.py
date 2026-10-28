"""FDB's test mask and the feasibility check that stands in for upstream's endless loop.

``check_bridge_feasible`` is a detector: it claims to know when ``create_mask`` would spin
forever. Each shape it rejects is planted here, and each verdict is checked against the
upstream function itself, run under a draw budget so a hang becomes a failure.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from spectramr_baselines._fdb_mask import (
    bridge_test_steps,
    check_bridge_feasible,
    last_round_radius,
    normal_density_mask,
    seeded_numpy,
)
from spectramr_baselines.fdb import _FDB_SCRIPT_UTIL, _ensure_upstream_on_sys_path

needs_fdb = pytest.mark.skipif(not _FDB_SCRIPT_UTIL.exists(), reason="FDB vendored upstream absent")

_SIZE, _STEPS = 32, 12


def _mask(sigma: float = 0.3, seed: int = 0) -> torch.Tensor:
    radius = last_round_radius(_SIZE, _SIZE, _STEPS, "singlecoil")
    return normal_density_mask(_SIZE, _SIZE, 4, sigma=sigma, centre_radius=radius, seed=seed)


@pytest.mark.parametrize(("rate", "steps"), [(4, 1500), (8, 1750)])
def test_the_step_formula_reproduces_the_readme(rate: int, steps: int) -> None:
    """``sample.py`` is run at 1500 steps for R=4 and 1750 for R=8, trained at R'=2."""
    assert bridge_test_steps(1000, 2, rate) == steps


def test_a_rate_of_one_is_refused() -> None:
    with pytest.raises(ValueError, match="must exceed 1"):
        bridge_test_steps(1000, 1, 4)


def test_the_mask_hits_the_acceleration_exactly_and_acquires_the_centre() -> None:
    mask = _mask()
    assert mask.dtype == torch.float32
    assert int(mask.sum()) == _SIZE * _SIZE // 4
    radius = last_round_radius(_SIZE, _SIZE, _STEPS, "singlecoil")
    x = torch.arange(_SIZE, dtype=torch.float64)[:, None] - _SIZE / 2
    y = torch.arange(_SIZE, dtype=torch.float64)[None, :] - _SIZE / 2
    assert bool(mask[(x**2 + y**2) < radius**2].all())


def test_the_mask_is_a_function_of_its_seed() -> None:
    assert torch.equal(_mask(seed=3), _mask(seed=3))
    assert not torch.equal(_mask(seed=3), _mask(seed=4))


def test_a_disk_larger_than_the_budget_is_refused() -> None:
    with pytest.raises(ValueError, match="more than R=4 allows"):
        normal_density_mask(_SIZE, _SIZE, 4, sigma=0.3, centre_radius=_SIZE, seed=0)


def test_seeded_numpy_restores_the_callers_stream() -> None:
    np.random.seed(123)
    expected = np.random.rand(3)
    np.random.seed(123)
    with seeded_numpy(0):
        np.random.rand(100)
    assert np.array_equal(np.random.rand(3), expected)


# --- planted violations, one per way the schedule can fail ----------------


def _hole_inside_last_radius() -> torch.Tensor:
    """The final round cannot reach a hole at the centre."""
    mask = _mask()
    mask[_SIZE // 2, _SIZE // 2] = 0
    return mask


def _holes_only_mid_band() -> torch.Tensor:
    """Early rounds, whose radius still covers the mid band, find nothing to remove."""
    x = torch.arange(_SIZE, dtype=torch.float64)[:, None] - _SIZE / 2
    y = torch.arange(_SIZE, dtype=torch.float64)[None, :] - _SIZE / 2
    d = (x**2 + y**2).sqrt()
    return (~((d > 6) & (d < 9))).to(torch.float32)


@pytest.mark.parametrize("planted", [_hole_inside_last_radius, _holes_only_mid_band])
def test_an_unfinishable_schedule_is_named(planted) -> None:
    with pytest.raises(ValueError, match="cannot finish"):
        check_bridge_feasible(planted(), _STEPS, "singlecoil")


def test_the_generated_mask_passes() -> None:
    check_bridge_feasible(_mask(), _STEPS, "singlecoil")


@needs_fdb
@pytest.mark.parametrize(
    ("build", "finishes"),
    [(_mask, True), (_hole_inside_last_radius, False), (_holes_only_mid_band, False)],
)
def test_the_verdict_matches_upstreams_create_mask(monkeypatch, build, finishes: bool) -> None:
    """Run the real ``create_mask`` with a draw budget: a hang must become the raise."""
    import importlib

    _ensure_upstream_on_sys_path()
    fdb = importlib.import_module("utils.fdb")
    bridge = fdb.DiffusionBridge(
        steps=_STEPS, undersampling_rate=2, image_size=_SIZE, data_type="singlecoil"
    )
    draws = {"n": 0}
    real = np.random.randint

    def bounded(*args, **kwargs):
        draws["n"] += 1
        if draws["n"] > 2_000_000:
            raise TimeoutError("create_mask did not finish")
        return real(*args, **kwargs)

    monkeypatch.setattr(np.random, "randint", bounded)
    mask = build()
    m_us = mask.expand(1, 2, -1, -1)
    if finishes:
        M = bridge.create_mask(m_us)
        assert np.array_equal(M[-1] == 0, mask.numpy() == 0), "every hole is removed by T"
    else:
        with pytest.raises(TimeoutError):
            bridge.create_mask(m_us)
