"""The traced Compose records the dataset output, then what each member changed."""

from __future__ import annotations

import pytest
import torch
import torchio as tio

from spectramr.core import snapshot_trace as st
from spectramr.data.transforms.traced_compose import TracedCompose, plane

pytestmark = pytest.mark.unit


def _subject() -> tio.Subject:
    return tio.Subject(
        input=tio.ScalarImage(tensor=torch.rand(2, 8, 8, 1)),
        target=tio.ScalarImage(tensor=torch.rand(1, 8, 8, 1)),
    )


def _scale_input(subject: tio.Subject) -> tio.Subject:
    subject["input"].set_data(subject["input"].data * 2)
    return subject


class _ScaleInput(tio.Transform):
    def apply_transform(self, subject):
        return _scale_input(subject)

    def applied_record(self):
        return {"factor": 2}


def test_outside_a_trace_it_is_a_plain_compose() -> None:
    out = TracedCompose([_ScaleInput()])(_subject())
    assert out["input"].data.shape == (2, 8, 8, 1)
    assert st.current() is None


def test_each_member_records_only_what_it_changed() -> None:
    trace = st.SnapshotTrace(phase="train", step=0, batched=False)
    with st.active(trace):
        TracedCompose([_ScaleInput(), tio.Lambda(lambda x: x)])(_subject())
    names = [s.name for s in trace.stages]
    assert names == ["dataset_output", "00__ScaleInput", "01_Lambda"]
    assert set(trace.stages[0].tensors) == {"input", "target"}
    scaled = trace.stages[1]
    assert set(scaled.tensors) == {"input"} and scaled.params["applied_record"] == {"factor": 2}
    assert scaled.tensors["input"].shape == (2, 8, 8)  # central plane of [C, H, W, D]
    assert trace.stages[2].params["changed"] == []


def test_the_builder_returns_a_traced_compose() -> None:
    from spectramr.data.builders.torchio_transform_builder import _compose_copying_once

    assert isinstance(_compose_copying_once([_ScaleInput()]), TracedCompose)


def test_plane_takes_the_central_depth_slice() -> None:
    t = torch.arange(12.0).view(1, 2, 2, 3)
    assert torch.equal(plane(t), t[..., 1])
    assert plane(torch.ones(2, 3, 3)).shape == (2, 3, 3)
