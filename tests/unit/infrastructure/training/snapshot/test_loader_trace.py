"""The loader re-fetch records the sample's loader stages and advances no training RNG."""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torchio as tio

from spectramr.core import snapshot_trace as st
from spectramr.data.transforms.traced_compose import TracedCompose
from spectramr.infrastructure.training.snapshot.loader_trace import source_dataset, trace_sample

pytestmark = pytest.mark.unit


class _Noise(tio.Transform):
    """Draws from every RNG a real transform might use."""

    def __init__(self) -> None:
        super().__init__()
        self.gen = torch.Generator().manual_seed(7)
        self.np_rng = np.random.default_rng(3)

    def apply_transform(self, subject):
        n = (
            torch.rand(1, generator=self.gen)
            + torch.rand(1)
            + self.np_rng.random()
            + np.random.rand()
        )
        subject["input"].set_data(subject["input"].data + float(n) + random.random())
        return subject


class _Dataset:
    def __init__(self) -> None:
        self.transform = TracedCompose([_Noise()])

    def __len__(self) -> int:
        return 3

    def __getitem__(self, i: int):
        return self.transform(tio.Subject(input=tio.ScalarImage(tensor=torch.zeros(1, 4, 4, 1))))


def _draws(ds: _Dataset) -> tuple:
    noise = ds.transform.transforms[0]
    return (
        torch.rand(1).item(),
        np.random.rand(),
        random.random(),
        torch.rand(1, generator=noise.gen).item(),
        noise.np_rng.random(),
    )


def test_every_random_stream_continues_as_if_nothing_ran() -> None:
    ds_a, ds_b = _Dataset(), _Dataset()
    torch.manual_seed(0), np.random.seed(0), random.seed(0)
    expected = _draws(ds_a)

    torch.manual_seed(0), np.random.seed(0), random.seed(0)
    trace = st.SnapshotTrace(phase="train", step=5)
    with st.active(trace):
        trace_sample(trace, ds_b, index=4, seed=5)
    assert _draws(ds_b) == expected
    assert trace.batched is True, "the step's own stages are still sliced to sample 0"


def test_the_loader_stages_land_in_the_trace_whole() -> None:
    trace = st.SnapshotTrace(phase="train", step=1)
    with st.active(trace):
        trace_sample(trace, _Dataset(), index=0, seed=1)
    assert [s.name for s in trace.stages] == ["dataset_output", "00__Noise"]
    assert trace.stages[1].tensors["input"].shape == (1, 4, 4)


def test_the_same_seed_reproduces_the_same_sample() -> None:
    outs = []
    for _ in range(2):
        trace = st.SnapshotTrace(phase="train", step=2)
        with st.active(trace):
            trace_sample(trace, _Dataset(), index=0, seed=2)
        outs.append(trace.stages[1].tensors["input"])
    assert torch.equal(*outs)


def test_a_patch_queue_is_traced_through_its_subjects_dataset() -> None:
    inner = _Dataset()
    assert source_dataset(SimpleNamespace(subjects_dataset=inner)) is inner
    assert source_dataset(inner) is inner
