"""Poisson-disk master-pattern caching.

``_generate_master_pattern`` is ~99.9% Python interpreter time and costs
0.58 s at 128² / 2.38 s at 256² / 4.05 s at 320². ``generate_batch_masks``
asks for a mask once per sample per step, so an uncached accelerator spends
seconds of blocking host time per training step.

The cache these pin was previously disabled with the rationale that it traded
"CPU computation for GPU memory savings (prevents OOM during long training
runs)". That rationale was false: the cached value is a host-side list of
``(y, x)`` index pairs and never touches the device. ``test_cache_holds_no
_device_tensors`` is the executable form of that claim, so it cannot be
reintroduced silently.
"""

from __future__ import annotations

import torch

from spectramr.infrastructure.physics.sampling import PoissonDiskKSpaceAccelerator

CPU = torch.device("cpu")


def test_cached_pattern_equals_regenerated_pattern() -> None:
    """Caching must return exactly what regeneration would have produced.

    With a seed the generator is a pure function of ``(height, width)``, so the
    regenerated pattern is the oracle for the cached one.
    """
    accel = PoissonDiskKSpaceAccelerator(num_timesteps=1000, seed=0)
    for size in (48, 64):
        regenerated = accel._generate_master_pattern(size, size, CPU)
        assert accel._generate_master_pattern(size, size, CPU) == regenerated
        assert accel._get_master_pattern(size, size, CPU) == regenerated


def test_repeated_masks_are_identical() -> None:
    """The per-sample call is what ``generate_batch_masks`` repeats."""
    accel = PoissonDiskKSpaceAccelerator(num_timesteps=1000, seed=0)
    first = accel.get_acceleration_mask((1, 64, 64), 500, CPU)
    for _ in range(3):
        assert torch.equal(accel.get_acceleration_mask((1, 64, 64), 500, CPU), first)


def test_cache_holds_no_device_tensors() -> None:
    """The disabled-cache rationale claimed GPU accumulation; refute it."""
    accel = PoissonDiskKSpaceAccelerator(num_timesteps=1000, seed=0)
    accel.get_acceleration_mask((1, 48, 48), 100, CPU)
    assert accel._cached_patterns, "the pattern should have been cached"
    for value in accel._cached_patterns.values():
        assert isinstance(value, list)
        assert not torch.is_tensor(value)
        assert all(isinstance(point, tuple) for point in value)


def test_cache_is_bounded() -> None:
    """Distinct shapes must not grow the cache without limit."""
    accel = PoissonDiskKSpaceAccelerator(num_timesteps=1000, seed=0)
    limit = accel._PATTERN_CACHE_MAX
    for size in range(24, 24 + 4 * (limit + 2), 4):
        accel._get_master_pattern(size, size, CPU)
    assert len(accel._cached_patterns) <= limit


def test_distinct_shapes_get_distinct_patterns() -> None:
    """A shape-keyed cache must not serve one shape's pattern for another."""
    accel = PoissonDiskKSpaceAccelerator(num_timesteps=1000, seed=0)
    small = accel._get_master_pattern(48, 48, CPU)
    large = accel._get_master_pattern(96, 96, CPU)
    assert small != large
    assert accel._get_master_pattern(48, 48, CPU) == small
