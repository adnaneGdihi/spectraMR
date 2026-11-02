"""Perf regression for NonCartesianSimulationTransform trajectory/DCF cache.

The trajectory and density-compensation function depend only on
``(pattern, im_size)`` and the fixed accel/kwargs, yet were regenerated on
every ``apply_transform`` (DataLoader worker hot path). They are now cached
alongside the NUFFT operators. This test pins that the cache is populated,
reused across items at a fixed shape, and rebuilt on a shape change — and
that the transform output is unchanged.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
tio = pytest.importorskip("torchio")
pytest.importorskip("torchkbnufft")

from spectramr.data.transforms.non_cartesian import (  # noqa: E402
    NonCartesianSimulationTransform,
)


def _subject(h: int = 32, w: int = 32) -> tio.Subject:
    return tio.Subject(input=tio.ScalarImage(tensor=torch.rand(1, h, w, 1)))


@pytest.mark.unit
def test_trajectory_and_dcf_cached_and_reused():
    t = NonCartesianSimulationTransform(pattern="radial", acceleration=8.0)
    assert t._cached_traj is None

    t(_subject(32, 32))
    assert t._cached_traj is not None and t._cached_dcf is not None
    assert t._cached_traj_key == ("radial", (32, 32))
    traj_obj, dcf_obj = t._cached_traj, t._cached_dcf

    # Same shape → same cached objects (no rebuild).
    t(_subject(32, 32))
    assert t._cached_traj is traj_obj
    assert t._cached_dcf is dcf_obj


@pytest.mark.unit
def test_trajectory_cache_rebuilds_on_shape_change():
    t = NonCartesianSimulationTransform(pattern="radial", acceleration=8.0)
    t(_subject(32, 32))
    first = t._cached_traj
    t(_subject(24, 24))
    assert t._cached_traj_key == ("radial", (24, 24))
    assert t._cached_traj is not first


@pytest.mark.unit
def test_output_deterministic_across_cached_calls():
    """Caching the trajectory must not change the (noise-free) output for the
    same input tensor."""
    base = torch.rand(1, 32, 32, 1)
    t = NonCartesianSimulationTransform(
        pattern="radial", acceleration=8.0, enable_noise=False
    )
    o1 = t(tio.Subject(input=tio.ScalarImage(tensor=base.clone())))["input"].data
    o2 = t(tio.Subject(input=tio.ScalarImage(tensor=base.clone())))["input"].data
    assert torch.allclose(o1, o2)


@pytest.mark.unit
def test_simulated_samples_survive_patch_sampling():
    """Keys the transform adds reach a training patch, and the copied target is an Image."""
    t = NonCartesianSimulationTransform(pattern="spiral", im_size=(32, 32))
    out = t(tio.Subject(input=tio.ScalarImage(tensor=torch.rand(1, 32, 32, 2))))
    patch = next(iter(tio.data.UniformSampler((32, 32, 1))(out)))
    assert {"measured_kspace", "trajectory", "dcf", "target"} <= set(patch.keys())
    assert isinstance(patch["target"], tio.Image)
