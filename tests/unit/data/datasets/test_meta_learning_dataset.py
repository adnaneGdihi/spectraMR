"""Unit tests for MetaLearningDataset.

Tests pure-logic portions using a tiny in-memory base dataset.
No external data files required.

Covers:
  - Construction with default parameters.
  - __len__ returns tasks_per_epoch.
  - __getitem__ returns dict with 'support', 'query', 'task_params'.
  - Support/query tuples have the right batch dimension.
  - _sample_task_params samples acceleration from config when configured.
  - _get_timestep_for_acceleration mapping (monotone, boundary clamp).
  - tasks_per_epoch parameter honoured.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from spectramr.data.datasets.meta_learning_dataset import MetaLearningDataset


# ── Minimal in-memory base dataset ────────────────────────────────────────────


class _TinyImageDataset(Dataset):
    """16 tiny (1, 8, 8) image tensors — no on-disk files."""

    def __init__(self, n: int = 16, size: int = 8) -> None:
        self.data = [torch.rand(1, size, size) for _ in range(n)]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return {"target": self.data[idx % len(self.data)]}


# ── Construction ──────────────────────────────────────────────────────────────


def test_construction_default_params() -> None:
    base = _TinyImageDataset(n=16)
    ds = MetaLearningDataset(base_dataset=base, support_size=2, query_size=2, tasks_per_epoch=5)
    assert len(ds) == 5


def test_construction_honoring_tasks_per_epoch() -> None:
    base = _TinyImageDataset(n=16)
    ds = MetaLearningDataset(base_dataset=base, support_size=2, query_size=2, tasks_per_epoch=10)
    assert len(ds) == 10


# ── __getitem__ structure ─────────────────────────────────────────────────────


def test_canary_getitem_returns_required_keys() -> None:
    base = _TinyImageDataset(n=16)
    ds = MetaLearningDataset(
        base_dataset=base,
        support_size=2,
        query_size=2,
        tasks_per_epoch=4,
    )
    item = ds[0]
    assert "support" in item
    assert "query" in item
    assert "task_params" in item


def test_getitem_support_query_are_tuple_of_two_tensors() -> None:
    base = _TinyImageDataset(n=16)
    ds = MetaLearningDataset(
        base_dataset=base,
        support_size=2,
        query_size=2,
        tasks_per_epoch=2,
    )
    item = ds[0]
    support_x, support_y = item["support"]
    query_x, query_y = item["query"]
    assert isinstance(support_x, torch.Tensor)
    assert isinstance(support_y, torch.Tensor)
    assert isinstance(query_x, torch.Tensor)
    assert isinstance(query_y, torch.Tensor)


def test_getitem_support_batch_dim_matches_support_size() -> None:
    base = _TinyImageDataset(n=16)
    ds = MetaLearningDataset(
        base_dataset=base,
        support_size=3,
        query_size=2,
        tasks_per_epoch=2,
    )
    support_x, _ = ds[0]["support"]
    assert support_x.shape[0] == 3


def test_getitem_query_batch_dim_matches_query_size() -> None:
    base = _TinyImageDataset(n=16)
    ds = MetaLearningDataset(
        base_dataset=base,
        support_size=2,
        query_size=4,
        tasks_per_epoch=2,
    )
    query_x, _ = ds[0]["query"]
    assert query_x.shape[0] == 4


# ── Task parameter sampling ───────────────────────────────────────────────────


def test_task_params_contains_acceleration_when_configured() -> None:
    base = _TinyImageDataset(n=16)
    ds = MetaLearningDataset(
        base_dataset=base,
        task_config={"acceleration_factors": [2, 4, 8]},
        support_size=2,
        query_size=2,
        tasks_per_epoch=4,
    )
    item = ds[0]
    assert "acceleration" in item["task_params"]
    assert item["task_params"]["acceleration"] in [2, 4, 8]


def test_task_params_empty_when_no_task_config() -> None:
    base = _TinyImageDataset(n=16)
    ds = MetaLearningDataset(base_dataset=base, support_size=2, query_size=2, tasks_per_epoch=2)
    item = ds[0]
    assert item["task_params"] == {}


# ── _get_timestep_for_acceleration ────────────────────────────────────────────


def test_timestep_min_acceleration_maps_to_zero() -> None:
    base = _TinyImageDataset(n=16)
    ds = MetaLearningDataset(base_dataset=base, support_size=2, query_size=2, tasks_per_epoch=2)
    assert ds._get_timestep_for_acceleration(1.0) == 0


def test_timestep_max_acceleration_maps_to_999() -> None:
    base = _TinyImageDataset(n=16)
    ds = MetaLearningDataset(base_dataset=base, support_size=2, query_size=2, tasks_per_epoch=2)
    assert ds._get_timestep_for_acceleration(64.0) == 999


def test_timestep_is_monotone() -> None:
    base = _TinyImageDataset(n=16)
    ds = MetaLearningDataset(base_dataset=base, support_size=2, query_size=2, tasks_per_epoch=2)
    ts_2 = ds._get_timestep_for_acceleration(2.0)
    ts_8 = ds._get_timestep_for_acceleration(8.0)
    ts_32 = ds._get_timestep_for_acceleration(32.0)
    assert ts_2 < ts_8 < ts_32


def test_timestep_clamps_below_zero() -> None:
    """Acceleration below 1.0 should clamp to timestep 0."""
    base = _TinyImageDataset(n=16)
    ds = MetaLearningDataset(base_dataset=base, support_size=2, query_size=2, tasks_per_epoch=2)
    assert ds._get_timestep_for_acceleration(0.0) == 0


def test_the_acceleration_mapping_and_the_mask_generator_share_one_ladder():
    """The two 1000s were independent literals; one named resolution keeps them agreed (#2443)."""
    from spectramr.data.datasets import meta_learning_dataset as module

    ds = MetaLearningDataset(base_dataset=_TinyImageDataset(), support_size=2, query_size=2, tasks_per_epoch=1)
    assert ds.mask_generator.num_timesteps == module.MASK_LADDER_STEPS
    assert ds._get_timestep_for_acceleration(64.0) == module.MASK_LADDER_STEPS - 1
    assert ds._get_timestep_for_acceleration(1.0) == 0
