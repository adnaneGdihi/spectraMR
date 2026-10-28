"""Unit tests for ``spectramr.data.builders.data_loader_builder``.

Focuses on ``wrap_with_distributed_sampler`` — the DDP rewrap helper that must
faithfully carry over the source loader's settings (CLAUDE.md pitfall #11 keeps
DataLoader construction inside the data-builders subtree).
"""

from __future__ import annotations

from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Sampler, TensorDataset

from spectramr.data.builders.data_loader_builder import wrap_with_distributed_sampler


class _SequentialFakeSampler(Sampler):
    """A real ``Sampler`` so ``DataLoader`` accepts it without a process group."""

    def __init__(self, dataset) -> None:
        self._n = len(dataset)

    def __iter__(self):
        return iter(range(self._n))

    def __len__(self) -> int:
        return self._n


def _patched_distributed_sampler(dataset):
    """Stub ``DistributedSampler`` to avoid initializing a process group."""
    return patch(
        "torch.utils.data.distributed.DistributedSampler",
        lambda ds, shuffle=False: _SequentialFakeSampler(ds),
    )


def test_distributed_rewrap_preserves_prefetch_factor() -> None:
    """The rewrapped DDP loader must keep the source loader's prefetch_factor.

    Regression: ``prefetch_factor`` was not forwarded, so a loader tuned with
    ``prefetch_factor=4`` silently fell back to the default (2) under DDP.
    """
    dataset = TensorDataset(torch.zeros(8, 2))
    loader = DataLoader(dataset, batch_size=2, num_workers=2, prefetch_factor=4)

    with _patched_distributed_sampler(dataset):
        rewrapped = wrap_with_distributed_sampler(loader)

    assert rewrapped.prefetch_factor == 4


def test_distributed_rewrap_preserves_core_settings() -> None:
    """Batch size, workers, pin-memory and drop-last carry across the rewrap."""
    dataset = TensorDataset(torch.zeros(8, 2))
    loader = DataLoader(
        dataset, batch_size=4, num_workers=2, drop_last=True, pin_memory=False
    )

    with _patched_distributed_sampler(dataset):
        rewrapped = wrap_with_distributed_sampler(loader)

    assert rewrapped.batch_size == 4
    assert rewrapped.num_workers == 2
    assert rewrapped.drop_last is True


# ── Dead-code prune pin (2026-06 data-pipeline audit) ──────────────────
# The duplicate ``DataLoaderBuilder`` / ``DataLoaderBuildConfig`` /
# ``build_data_loaders`` trio had ZERO production callers (the live
# train/val loader path is ``infrastructure/training/builders/DataBuilder``
# + ``DataPipelineDirector``; the same-named fluent builder in
# ``infrastructure/builders/leaf/data_builders.py`` is a different class).
# Only ``wrap_with_distributed_sampler`` is consumed (pipelines/parallel.py).
# These tests pin the deletion so the dead surface cannot silently return.


def test_dead_loader_trio_removed_from_module() -> None:
    import spectramr.data.builders.data_loader_builder as mod

    for name in ("DataLoaderBuilder", "DataLoaderBuildConfig", "build_data_loaders"):
        assert not hasattr(mod, name), (
            f"dead symbol {name!r} re-introduced in data_loader_builder — the "
            "live loader path is infrastructure/training/builders/DataBuilder"
        )


def test_dead_loader_trio_removed_from_package_exports() -> None:
    import spectramr.data.builders as pkg

    for name in ("DataLoaderBuilder", "DataLoaderBuildConfig"):
        assert name not in pkg.__all__
        assert not hasattr(pkg, name)


def test_wrap_with_distributed_sampler_still_importable() -> None:
    from spectramr.data.builders.data_loader_builder import (
        wrap_with_distributed_sampler as fn,
    )

    assert callable(fn)


def test_distributed_rewrap_preserves_generator_and_timeout() -> None:
    """These were dropped by the rewrap, so DDP quietly changed the loader."""
    dataset = TensorDataset(torch.zeros(8, 2))
    generator = torch.Generator().manual_seed(7)
    loader = DataLoader(
        dataset, batch_size=2, num_workers=2, timeout=30, generator=generator, pin_memory=False
    )

    with _patched_distributed_sampler(dataset):
        rewrapped = wrap_with_distributed_sampler(loader)

    assert rewrapped.generator is generator
    assert rewrapped.timeout == 30


def test_distributed_rewrap_refuses_a_custom_batch_sampler() -> None:
    """A batch_sampler loader has batch_size None; rebuilding it would drop the batching rule."""
    import pytest
    from torch.utils.data import BatchSampler, SequentialSampler

    dataset = TensorDataset(torch.zeros(8, 2))
    loader = DataLoader(dataset, batch_sampler=BatchSampler(SequentialSampler(dataset), 2, False))

    with pytest.raises(ValueError, match="batch_sampler"):
        wrap_with_distributed_sampler(loader)


def test_distributed_rewrap_says_when_it_replaces_a_custom_sampler(caplog) -> None:
    """The volume-blocked slice order is lost under DDP; that must not be silent."""
    dataset = TensorDataset(torch.zeros(8, 2))
    loader = DataLoader(dataset, batch_size=2, sampler=_SequentialFakeSampler(dataset))

    with _patched_distributed_sampler(dataset), caplog.at_level("WARNING"):
        wrap_with_distributed_sampler(loader)

    assert "_SequentialFakeSampler" in caplog.text


def test_distributed_rewrap_keeps_quiet_for_the_default_samplers(caplog) -> None:
    dataset = TensorDataset(torch.zeros(8, 2))
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    with _patched_distributed_sampler(dataset), caplog.at_level("WARNING"):
        wrap_with_distributed_sampler(loader)

    assert "replacing the loader" not in caplog.text


def test_distributed_rewrap_shards_the_volume_blocked_sampler_instead_of_replacing_it(
    caplog,
) -> None:
    """The blocked order is what keeps slice-expanded volumes affordable, so DDP keeps it."""
    import itertools

    from spectramr.data.samplers import VolumeBlockedSliceSampler

    volumes = [f"g{g}_f{i}" for g in range(4) for i in range(3)]
    container_volumes = [
        frozenset(p)
        for g in range(4)
        for p in itertools.permutations(volumes[g * 3 : g * 3 + 3], 2)
    ]
    index_map = [(c, s) for c in range(len(container_volumes)) for s in range(4)]
    dataset = TensorDataset(torch.zeros(len(index_map), 2))
    source = VolumeBlockedSliceSampler(index_map, container_volumes, max_resident_volumes=4)
    loader = DataLoader(dataset, batch_size=2, sampler=source)

    with (
        patch("torch.distributed.get_rank", return_value=1),
        patch("torch.distributed.get_world_size", return_value=2),
        caplog.at_level("WARNING"),
    ):
        rewrapped = wrap_with_distributed_sampler(loader)

    assert isinstance(rewrapped.sampler, VolumeBlockedSliceSampler)
    assert rewrapped.sampler is not source
    assert len(rewrapped.sampler) < len(source)  # a share, not the whole epoch
    assert len(rewrapped) == (len(rewrapped.sampler) + 1) // 2
    assert "replacing the loader" not in caplog.text
