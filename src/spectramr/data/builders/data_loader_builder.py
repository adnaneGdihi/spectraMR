"""DDP loader rewrap helper (data-builders SSOT for DataLoader construction).

This module used to also house a ``DataLoaderBuilder`` / ``DataLoaderBuildConfig``
/ ``build_data_loaders`` trio. That surface had **zero production callers** —
the live train/val loader path is ``infrastructure/training/builders/DataBuilder``
via ``DataPipelineDirector``, and the same-named fluent builder in
``infrastructure/builders/leaf/data_builders.py`` is a different class — so it
was pruned in the 2026-06 data-pipeline audit (deletion pinned by
``tests/unit/data/builders/test_data_loader_builder.py``). Its ``shuffle`` /
``drop_last`` knobs were also unread (the methods hardcoded their own values),
making them pitfall-#15 silent no-ops.

Only :func:`wrap_with_distributed_sampler` is consumed (``pipelines/parallel.py``).
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — annotation-only import
    import torch


def wrap_with_distributed_sampler(
    loader: "torch.utils.data.DataLoader",
    *,
    shuffle: bool = False,
) -> "torch.utils.data.DataLoader":
    """Return a new ``DataLoader`` that wraps ``loader``'s dataset with a ``DistributedSampler``.

    PyTorch does not expose a public way to swap a sampler on an existing
    ``DataLoader``, so reconstructing the loader is the canonical pattern.
    This helper lives in the data-builders subtree (CLAUDE.md pitfall #11)
    so pipelines code can wrap loaders for DDP without instantiating
    ``DataLoader`` directly. Phase 4 of
    ``TODO/backlog_ssot_and_layering_cleanup.md``.

    Args:
        loader: The non-distributed loader to rewrap.
        shuffle: Whether the distributed sampler should shuffle.

    Returns:
        A new ``DataLoader`` with ``DistributedSampler`` attached, preserving
        the source loader's batch size, workers, pin-memory, drop-last,
        collate, worker-init, and persistent-workers settings.
    """
    import logging

    from torch.utils.data import DataLoader as _DataLoader
    from torch.utils.data import RandomSampler, SequentialSampler
    from torch.utils.data.distributed import DistributedSampler

    if loader.batch_size is None:
        raise ValueError(
            "wrap_with_distributed_sampler cannot rewrap a loader built with a custom "
            "batch_sampler (batch_size is None): rebuilding it with a DistributedSampler "
            "would drop the batching rule. Shard the batch_sampler instead."
        )
    from spectramr.data.samplers import VolumeBlockedSliceSampler

    if isinstance(loader.sampler, VolumeBlockedSliceSampler):
        # Keep the volume-blocked order: it is what makes slice-expanded volumes
        # affordable (~240x fewer decodes than a plain shuffle). Each rank gets a
        # static share of the containers, so a volume is decoded on one rank rather
        # than on every rank.
        import torch.distributed as dist

        sampler = loader.sampler.sharded(dist.get_rank(), dist.get_world_size())
    else:
        if not isinstance(loader.sampler, (RandomSampler, SequentialSampler)):
            logging.getLogger(__name__).warning(
                "[DDP] replacing the loader's %s with a DistributedSampler; its ordering "
                "is not preserved across ranks.",
                type(loader.sampler).__name__,
            )
        sampler = DistributedSampler(loader.dataset, shuffle=shuffle)
    return _DataLoader(
        dataset=loader.dataset,
        batch_size=loader.batch_size,
        sampler=sampler,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        collate_fn=loader.collate_fn,
        worker_init_fn=loader.worker_init_fn,
        # Preserve the source loader's prefetch depth; without this the rewrap
        # silently reverts to PyTorch's default (2). prefetch_factor must be
        # None when there are no workers.
        prefetch_factor=(loader.prefetch_factor if loader.num_workers > 0 else None),
        persistent_workers=(loader.persistent_workers if loader.num_workers > 0 else False),
        generator=loader.generator,
        timeout=loader.timeout,
        multiprocessing_context=loader.multiprocessing_context,
        pin_memory_device=loader.pin_memory_device,
    )
