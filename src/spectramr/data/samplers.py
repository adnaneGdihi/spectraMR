"""Epoch-ordering samplers for slice-expanded volumetric datasets.

Canonical home for ``torch.utils.data.Sampler`` implementations that exist to make an
IO pattern affordable, as opposed to the ``torchio`` patch samplers built in
:mod:`spectramr.data.builders.sampler_factory` (which choose WHERE in a volume to sample).
These choose the ORDER in which already-chosen samples are visited.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence

import torch
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class VolumeBlockedSliceSampler(Sampler[int]):
    """Visit slice samples in volume-resident blocks, shuffled inside each block.

    **The problem.** Expanding a paired volumetric dataset to one sample per depth slice
    is the right sampling — a central-slice dataset throws away ~363/364 of every volume
    — but it makes the naive epoch order pathological. Under a plain shuffled loader each
    ``(container, slice)`` is visited once per epoch in random order, so a resident-volume
    budget only helps if it can hold the WHOLE working set. For the MRIxFields corpus that
    is ~45 volumes at ~226 MB, so every sample is very likely a fresh full-volume decode.
    Measured: one decode is ~0.51 s, which at 150k iterations is tens of GPU-hours of pure
    data wait per arm — the starvation failure recorded in
    ``docs/mrixfields_gpu_starvation_fix.rst``.

    **The fix** is not a bigger cache, it is a better order. Consume the containers that
    share a resident volume set together, so one decode is amortised over every foreground
    slice of every container that volume participates in. Volume sharing is high because
    the pairing policies are combinatorial: a 5-field ``all_pairs`` group is 20 ordered
    pairs drawn from 5 volumes, so 5 decodes serve ~5,000 samples. Simulated on a
    3-group/5-field/250-slice corpus with a 6-volume budget: a plain shuffle costs ~18,700
    decodes per epoch, this order costs ~78 — a 240x reduction.

    **Why not ``tio.Queue``.** This is precisely ``tio.Queue``'s algorithm — load a few
    volumes, emit many patches from each, shuffle them in a bounded buffer — and that was
    the first thing checked. Two things rule the Queue itself out here:

    1. ``MRIxFieldsPairedDataset`` emits plain dicts, not ``tio.Subject``. The
       ``DataPipelineDirector`` already routes ``mrixfields`` and ``npy_slice`` around the
       Queue for exactly this reason: the queue's patch-compatibility filter calls
       ``subj.get_images_names()`` and crashes on a dict.
    2. More fundamentally, ``tio.Queue`` amortises per *Subject*, and a Subject here would
       have to be a PAIR. A volume shared by N pairs would be decoded once per pair —
       40 volume-loads for a 20-pair group where 5 suffice. The sharing structure this
       sampler exploits is invisible at Subject granularity.

    So the algorithm is reused and the container is not. ``max_resident_volumes`` plays
    ``max_length``'s role (the RAM bound) and the block's slice count plays
    ``samples_per_volume``'s (the amortisation factor), except that both are derived from
    the data rather than guessed.

    **Shuffling.** Blocks are shuffled, and samples are shuffled WITHIN a block, so batches
    mix slices and containers freely. What they do not mix is volumes from different
    blocks — a block is a group of co-resident volumes, typically one subject-contrast
    group's whole field ladder. Batch statistics therefore see many slices and several
    field pairings but a restricted subject set within any short window. That is the
    deliberate trade for making the sampling affordable at all; the alternative on this
    corpus is not "better shuffling", it is the central slice.

    Args:
        index_map: the dataset's ``(container_idx, slice_idx)`` list, positionally
            aligned with dataset indices.
        container_volumes: volume paths each container reads, from
            ``MRIxFieldsPairedDataset.container_volume_paths()``.
        max_resident_volumes: how many decoded volumes a worker may hold — the RAM bound.
            Must be at least the widest container's volume count or a block cannot be
            formed; that case raises rather than silently thrashing.
        shuffle: ``False`` gives a deterministic volume-major order (the validation case:
            reproducible, and it keeps a capped/visualised prefix stable across checks).
        seed: base seed; the epoch is mixed in via :meth:`set_epoch` so each epoch differs
            while staying reproducible.
        rank, num_replicas: this process's place in a data-parallel group. With
            ``num_replicas > 1`` the sampler owns a static share of the containers: see
            :meth:`sharded`.
    """

    def __init__(
        self,
        index_map: Sequence[tuple[int, int]],
        container_volumes: Sequence[frozenset[str]],
        *,
        max_resident_volumes: int,
        shuffle: bool = True,
        seed: int = 0,
        rank: int = 0,
        num_replicas: int = 1,
    ) -> None:
        super().__init__()
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError(f"rank={rank} is outside num_replicas={num_replicas}.")
        if len(container_volumes) == 0:
            raise ValueError(
                "VolumeBlockedSliceSampler needs per-container volume paths; got none. "
                "It applies only to slice_mode='all_slices' — for 'central'/'volume' use "
                "the loader's ordinary shuffle."
            )
        widest = max(len(v) for v in container_volumes)
        if max_resident_volumes < widest:
            raise ValueError(
                f"max_resident_volumes={max_resident_volumes} is smaller than the widest "
                f"container's volume count ({widest}), so no block can be formed and "
                "every sample would evict the volume the next one needs. Raise the budget "
                "to at least the container arity (N sources + target)."
            )
        self._index_map = list(index_map)
        self._container_volumes = list(container_volumes)
        self._max_resident = int(max_resident_volumes)
        self._shuffle = bool(shuffle)
        self._seed = int(seed)
        self._epoch = 0

        # container -> the dataset indices that read it. Built once; the epoch shuffle
        # permutes blocks and their contents, never rebuilds this.
        by_container: dict[int, list[int]] = {}
        for dataset_idx, (container_idx, _slice_idx) in enumerate(self._index_map):
            by_container.setdefault(container_idx, []).append(dataset_idx)
        self._by_container = by_container

        # The containers this process visits, and how many samples each rank owns.
        # Static (epoch-independent) so ``__len__`` is a constant and every rank can
        # derive the whole assignment locally with no communication.
        self._rank, self._num_replicas = int(rank), int(num_replicas)
        shares = self._partition(self._num_replicas)
        self._owned = shares[self._rank]
        share_sizes = [sum(len(by_container.get(c, ())) for c in share) for share in shares]
        if min(share_sizes) == 0:
            # Containers with no slices count toward the split but add no samples, so a
            # rank can own containers and still have nothing to repeat up to the longest
            # share; refuse here instead of failing inside the first epoch.
            raise ValueError(
                f"rank {share_sizes.index(0)} of {self._num_replicas} would own no slice "
                f"samples (per-rank sample counts: {share_sizes}); use fewer ranks or a "
                "dataset with more non-empty containers."
            )
        self._local_samples = share_sizes[self._rank]
        self._target_len = max(share_sizes)
        # Share of the epoch's samples that are repeats made to equalise rank lengths.
        self._padded_fraction = 1.0 - sum(share_sizes) / (len(share_sizes) * self._target_len)

    def _partition(self, num_replicas: int) -> list[list[int]]:
        """Split the containers across ranks, keeping volume-sharing containers together.

        Containers that share a volume are one connected component, and a component is
        placed on one rank so each volume is decoded by one process instead of N. The
        components go to ranks greedily, heaviest first, to balance sample counts. With
        fewer components than ranks that cannot give every rank work, so the split falls
        back to single containers: correct, at the price of some volumes being decoded on
        more than one rank.
        """
        count = len(self._container_volumes)
        if num_replicas == 1:
            return [list(range(count))]
        if count < num_replicas:
            raise ValueError(
                f"{count} container(s) cannot be split across {num_replicas} ranks; a rank "
                "would have nothing to visit and its peers would wait on it."
            )
        parent = list(range(count))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        first_owner: dict[str, int] = {}
        for container_idx, volumes in enumerate(self._container_volumes):
            for volume in volumes:
                other = first_owner.setdefault(volume, container_idx)
                parent[find(container_idx)] = find(other)
        groups: dict[int, list[int]] = {}
        for container_idx in range(count):
            groups.setdefault(find(container_idx), []).append(container_idx)
        units = list(groups.values())
        if len(units) < num_replicas:
            units = [[container_idx] for container_idx in range(count)]

        weight = lambda unit: sum(len(self._by_container.get(c, ())) for c in unit)  # noqa: E731
        shares: list[list[int]] = [[] for _ in range(num_replicas)]
        load = [0] * num_replicas
        for unit in sorted(units, key=lambda u: (-weight(u), u[0])):
            target = min(range(num_replicas), key=lambda r: (load[r], r))
            shares[target].extend(unit)
            load[target] += weight(unit)
        return [sorted(share) for share in shares]

    def sharded(self, rank: int, num_replicas: int) -> VolumeBlockedSliceSampler:
        """The same sampler restricted to ``rank``'s share of an ``num_replicas`` group.

        Every rank's length is padded to the longest share by repeating its own samples,
        as ``DistributedSampler`` does, because FSDP and ZeRO-3 forwards are collectives
        and ranks that disagree on batch count would hang.
        """
        return VolumeBlockedSliceSampler(
            self._index_map,
            self._container_volumes,
            max_resident_volumes=self._max_resident,
            shuffle=self._shuffle,
            seed=self._seed,
            rank=rank,
            num_replicas=num_replicas,
        )

    def set_epoch(self, epoch: int) -> None:
        """Re-seed the shuffle for ``epoch`` (mirrors ``DistributedSampler.set_epoch``)."""
        self._epoch = int(epoch)

    def _blocks(self, order: list[int]) -> list[list[int]]:
        """Greedily pack containers into blocks whose volume union fits the budget.

        Greedy over the (possibly shuffled) container order rather than a globally optimal
        packing: optimal set-cover is not worth solving here, and the combinatorial
        policies already put the containers that share volumes adjacent to one another in
        manifest order.
        """
        blocks: list[list[int]] = []
        current: list[int] = []
        resident: set[str] = set()
        for container_idx in order:
            volumes = self._container_volumes[container_idx]
            if current and len(resident | volumes) > self._max_resident:
                blocks.append(current)
                current, resident = [], set()
            current.append(container_idx)
            resident |= volumes
        if current:
            blocks.append(current)
        return blocks

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self._seed + self._epoch)

        order = list(self._owned)
        if self._shuffle:
            order = [order[i] for i in torch.randperm(len(order), generator=generator)]

        blocks = self._blocks(order)
        if self._shuffle:
            blocks = [blocks[i] for i in torch.randperm(len(blocks), generator=generator)]

        pad = self._target_len - self._local_samples
        emitted: list[int] = []
        for block in blocks:
            samples: list[int] = []
            for container_idx in block:
                samples.extend(self._by_container.get(container_idx, []))
            if self._shuffle and samples:
                samples = [samples[i] for i in torch.randperm(len(samples), generator=generator)]
            if pad > 0:
                emitted.extend(samples)
            yield from samples
        # A shorter share repeats its own leading samples up to the longest share.
        for i in range(pad):
            yield emitted[i % len(emitted)]

    def __len__(self) -> int:
        return self._target_len

    def describe(self) -> str:
        """One-line amortisation summary, for the loader to log.

        Stamps the resolved behaviour into the run's output rather than leaving the
        operator to infer it: an ordering whose whole purpose is IO cost should say what
        that cost became.
        """
        blocks = self._blocks(list(self._owned))
        decodes = sum(len({v for c in b for v in self._container_volumes[c]}) for b in blocks)
        samples = self._local_samples
        per_decode = samples / decodes if decodes else 0.0
        return (
            f"VolumeBlockedSliceSampler: {samples} slice samples over {len(blocks)} "
            f"block(s), {decodes} volume decode(s) per epoch "
            f"(~{per_decode:.0f} samples per decode), "
            f"max_resident_volumes={self._max_resident}"
            + (
                f", {self._padded_fraction:.0%} of each epoch is repeated samples that "
                f"equalise {self._num_replicas} unequal rank shares"
                if self._padded_fraction > 0
                else ""
            )
        )
