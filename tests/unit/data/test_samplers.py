"""Tests for VolumeBlockedSliceSampler — the epoch order that makes all_slices affordable.

The sampler exists for an IO property, so the tests assert that property directly
(decodes per epoch under a simulated LRU) rather than only that it emits indices. A
sampler that yielded a correct permutation in a pathological order would satisfy every
coverage assertion and still reinstate the starvation it was written to prevent.
"""

from __future__ import annotations

import itertools
from collections import OrderedDict

import pytest

from spectramr.data.samplers import VolumeBlockedSliceSampler


def _corpus(groups: int = 3, fields: int = 5, slices: int = 20):
    """all_pairs over `groups` subject-contrast groups: the real container topology.

    Combinatorial pairing is what gives the sampler its leverage — a 5-field group is
    20 ordered pairs drawn from 5 volumes — so the fixture reproduces that rather than
    a flat list of independent items.
    """
    container_volumes: list[frozenset[str]] = []
    for g in range(groups):
        vols = [f"g{g}_f{i}" for i in range(fields)]
        container_volumes.extend(frozenset(p) for p in itertools.permutations(vols, 2))
    index_map = [(c, s) for c in range(len(container_volumes)) for s in range(slices)]
    return index_map, container_volumes


def _decodes(order, index_map, container_volumes, budget: int) -> int:
    """Volume decodes an LRU of `budget` would perform for this visit order."""
    lru: OrderedDict[str, int] = OrderedDict()
    misses = 0
    for dataset_idx in order:
        container_idx, _ = index_map[dataset_idx]
        for volume in container_volumes[container_idx]:
            if volume in lru:
                lru.move_to_end(volume)
            else:
                misses += 1
                lru[volume] = 1
                if len(lru) > budget:
                    lru.popitem(last=False)
    return misses


class TestCoverage:
    def test_yields_every_sample_exactly_once(self) -> None:
        index_map, container_volumes = _corpus()
        sampler = VolumeBlockedSliceSampler(
            index_map, container_volumes, max_resident_volumes=6
        )
        order = list(sampler)
        assert sorted(order) == list(range(len(index_map)))
        assert len(sampler) == len(index_map)

    def test_shuffled_and_unshuffled_cover_the_same_set(self) -> None:
        index_map, container_volumes = _corpus()
        kwargs = {"max_resident_volumes": 6}
        a = list(
            VolumeBlockedSliceSampler(
                index_map, container_volumes, shuffle=True, **kwargs
            )
        )
        b = list(
            VolumeBlockedSliceSampler(
                index_map, container_volumes, shuffle=False, **kwargs
            )
        )
        assert sorted(a) == sorted(b)
        assert a != b, "shuffle=True produced the deterministic order"


class TestTheIOPropertyItExistsFor:
    def test_blocked_order_needs_far_fewer_decodes_than_a_plain_shuffle(self) -> None:
        """THE reason the sampler exists. Measured, not asserted qualitatively.

        A plain shuffled epoch makes nearly every sample a fresh ~226 MB decode against
        a working set no budget can hold; the blocked order amortises one decode over
        every slice of every container that volume takes part in.
        """
        import random

        index_map, container_volumes = _corpus()
        budget = 6

        naive = list(range(len(index_map)))
        random.Random(0).shuffle(naive)
        blocked = list(
            VolumeBlockedSliceSampler(
                index_map, container_volumes, max_resident_volumes=budget
            )
        )

        naive_decodes = _decodes(naive, index_map, container_volumes, budget)
        blocked_decodes = _decodes(blocked, index_map, container_volumes, budget)

        # An order-of-magnitude bound, not the exact ratio: the point is the regime
        # change, and pinning the precise number would make the test brittle against
        # a harmless packing tweak.
        assert blocked_decodes * 10 < naive_decodes

    def test_decodes_approach_the_number_of_distinct_volumes(self) -> None:
        """The floor: with a budget covering a group, each volume is decoded ~once."""
        index_map, container_volumes = _corpus(groups=1, fields=5, slices=20)
        distinct = len({v for f in container_volumes for v in f})
        order = list(
            VolumeBlockedSliceSampler(
                index_map, container_volumes, max_resident_volumes=6
            )
        )
        assert _decodes(order, index_map, container_volumes, 6) <= distinct * 2

    def test_describe_reports_the_amortisation(self) -> None:
        index_map, container_volumes = _corpus(groups=1)
        sampler = VolumeBlockedSliceSampler(
            index_map, container_volumes, max_resident_volumes=6
        )
        text = sampler.describe()
        assert "samples per decode" in text
        assert "max_resident_volumes=6" in text


class TestPreconditionsFailLoud:
    def test_a_budget_below_the_widest_container_raises(self) -> None:
        """No silent thrash: below the container arity every item evicts the next one's
        volume, so the mode would 'work' at a hit rate of zero (pitfall #9)."""
        index_map, container_volumes = _corpus()
        with pytest.raises(ValueError, match="smaller than the widest container"):
            VolumeBlockedSliceSampler(
                index_map, container_volumes, max_resident_volumes=1
            )

    def test_no_container_volumes_raises(self) -> None:
        with pytest.raises(ValueError, match="needs per-container volume paths"):
            VolumeBlockedSliceSampler([], [], max_resident_volumes=6)


class TestDeterminism:
    def test_same_seed_and_epoch_gives_the_same_order(self) -> None:
        index_map, container_volumes = _corpus()
        kwargs = {"max_resident_volumes": 6, "seed": 7}
        a = list(VolumeBlockedSliceSampler(index_map, container_volumes, **kwargs))
        b = list(VolumeBlockedSliceSampler(index_map, container_volumes, **kwargs))
        assert a == b

    def test_set_epoch_changes_the_order_and_keeps_coverage(self) -> None:
        index_map, container_volumes = _corpus()
        sampler = VolumeBlockedSliceSampler(
            index_map, container_volumes, max_resident_volumes=6, seed=7
        )
        first = list(sampler)
        sampler.set_epoch(1)
        second = list(sampler)
        assert first != second
        assert sorted(first) == sorted(second)


def _shard(index_map, container_volumes, world: int, **kw):
    return [
        VolumeBlockedSliceSampler(
            index_map, container_volumes, max_resident_volumes=6, rank=r, num_replicas=world, **kw
        )
        for r in range(world)
    ]


class TestSharding:
    """A data-parallel group must see every sample, once, in equal numbers per rank.

    Equal counts are not cosmetic: FSDP and ZeRO-3 forwards are collectives, so ranks that
    disagree on batch count hang. The IO property is the other half: sharding must not
    scatter a volume's containers across ranks and bring the decode cost back.
    """

    def test_ranks_partition_the_samples_and_pad_to_equal_length(self) -> None:
        index_map, container_volumes = _corpus(groups=3)
        shards = _shard(index_map, container_volumes, 2)
        orders = [list(s) for s in shards]

        assert len({len(o) for o in orders}) == 1, "every rank must iterate the same count"
        assert [len(s) for s in shards] == [len(o) for o in orders]
        # Padding repeats a rank's own samples, so the ranks' sets stay disjoint.
        assert set(orders[0]).isdisjoint(orders[1])
        assert set(orders[0]) | set(orders[1]) == set(range(len(index_map)))

    def test_planted_uneven_split_is_padded_not_truncated(self) -> None:
        """3 groups over 2 ranks is 2 + 1: the short rank repeats, nobody drops a sample."""
        index_map, container_volumes = _corpus(groups=3, slices=4)
        shards = _shard(index_map, container_volumes, 2)
        assert len(shards[0]) == len(shards[1]) == 2 * 20 * 4
        owned = [len(set(s)) for s in map(list, shards)]
        assert sorted(owned) == [20 * 4, 2 * 20 * 4]

    def test_describe_states_how_much_of_the_epoch_is_padding(self) -> None:
        """2 + 1 groups: 240 distinct samples fill 320 slots, so a quarter is repeats."""
        index_map, container_volumes = _corpus(groups=3, slices=4)
        shards = _shard(index_map, container_volumes, 2)
        assert "25% of each epoch is repeated samples" in shards[0].describe()
        balanced = _shard(*_corpus(groups=4, slices=4), 2)
        assert "repeated samples" not in balanced[0].describe()

    def test_a_rank_with_no_samples_is_refused_at_construction(self) -> None:
        """A container with no slices counts toward the split but gives its rank nothing to
        repeat, which used to surface as a ZeroDivisionError inside the first epoch."""
        index_map = [(0, s) for s in range(4)]
        container_volumes = [{"a"}, {"b"}]
        with pytest.raises(ValueError, match="would own no slice samples"):
            _shard(index_map, container_volumes, 2)

    def test_a_volume_is_decoded_on_one_rank_when_groups_allow(self) -> None:
        index_map, container_volumes = _corpus(groups=4)
        shards = _shard(index_map, container_volumes, 2)
        per_rank = [
            {v for i in set(s) for v in container_volumes[index_map[i][0]]} for s in shards
        ]
        assert per_rank[0].isdisjoint(per_rank[1])

    def test_sharding_keeps_the_io_property(self) -> None:
        index_map, container_volumes = _corpus(groups=4)
        whole = _decodes(
            list(VolumeBlockedSliceSampler(index_map, container_volumes, max_resident_volumes=6)),
            index_map,
            container_volumes,
            budget=6,
        )
        sharded = sum(
            _decodes(list(s), index_map, container_volumes, budget=6)
            for s in _shard(index_map, container_volumes, 2)
        )
        # The total across ranks is the unsharded cost (each volume decoded once), not N times it.
        assert sharded <= whole + 2 * 5
        import random

        shuffled = list(range(len(index_map)))
        random.Random(0).shuffle(shuffled)
        plain = _decodes(shuffled, index_map, container_volumes, budget=6)
        assert sharded < plain / 10

    def test_fewer_groups_than_ranks_falls_back_to_splitting_containers(self) -> None:
        index_map, container_volumes = _corpus(groups=1)
        shards = _shard(index_map, container_volumes, 4)
        orders = [list(s) for s in shards]
        assert len({len(o) for o in orders}) == 1
        assert set().union(*map(set, orders)) == set(range(len(index_map)))

    def test_more_ranks_than_containers_is_refused(self) -> None:
        index_map, container_volumes = _corpus(groups=1, fields=2)  # 2 containers
        with pytest.raises(ValueError, match="cannot be split"):
            VolumeBlockedSliceSampler(
                index_map, container_volumes, max_resident_volumes=2, rank=0, num_replicas=3
            )

    def test_rank_outside_the_group_is_refused(self) -> None:
        index_map, container_volumes = _corpus()
        with pytest.raises(ValueError, match="outside num_replicas"):
            VolumeBlockedSliceSampler(
                index_map, container_volumes, max_resident_volumes=6, rank=2, num_replicas=2
            )

    def test_set_epoch_reorders_but_never_reassigns(self) -> None:
        index_map, container_volumes = _corpus(groups=4)
        shard = _shard(index_map, container_volumes, 2)[0]
        first = list(shard)
        shard.set_epoch(1)
        second = list(shard)
        assert first != second
        assert set(first) == set(second)

    def test_sharded_is_the_same_as_constructing_with_a_rank(self) -> None:
        index_map, container_volumes = _corpus(groups=4)
        base = VolumeBlockedSliceSampler(index_map, container_volumes, max_resident_volumes=6)
        direct = _shard(index_map, container_volumes, 2)[1]
        assert list(base.sharded(1, 2)) == list(direct)

    def test_every_rank_derives_the_same_assignment(self) -> None:
        """No communication: each rank computes the whole split locally."""
        index_map, container_volumes = _corpus(groups=5)
        a = [list(s) for s in _shard(index_map, container_volumes, 3)]
        b = [list(s) for s in _shard(index_map, container_volumes, 3)]
        assert a == b

    def test_describe_reports_the_ranks_own_share(self) -> None:
        index_map, container_volumes = _corpus(groups=4)
        text = _shard(index_map, container_volumes, 2)[0].describe()
        assert f"{len(index_map) // 2} slice samples" in text
