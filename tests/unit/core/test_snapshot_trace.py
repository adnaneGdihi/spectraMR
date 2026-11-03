"""The trace records only while active, keeps sample 0, and caps repeated stages."""

from __future__ import annotations

import pytest
import torch

from spectramr.core import snapshot_trace as st

pytestmark = pytest.mark.unit


def test_record_is_a_no_op_without_an_active_trace() -> None:
    assert st.current() is None
    st.record("batch", "input_raw", {"input": torch.ones(2, 1, 4, 4)})
    assert st.current() is None


def test_batched_trace_keeps_sample_zero_on_the_host() -> None:
    trace = st.SnapshotTrace(phase="train", step=3)
    with st.active(trace):
        st.record(
            "prepare", "input_prepared", {"x": torch.arange(32.0).view(2, 1, 4, 4), "lr": 0.5}
        )
    (stage,) = trace.stages
    assert stage.tensors["x"].shape == (1, 1, 4, 4)
    assert stage.tensors["x"].device.type == "cpu"
    assert stage.params == {"lr": 0.5}
    assert stage.source.endswith(":test_batched_trace_keeps_sample_zero_on_the_host")
    assert st.current() is None


def test_recorded_tensor_is_a_copy() -> None:
    x = torch.zeros(1, 1, 2, 2)
    trace = st.SnapshotTrace(phase="train", step=0)
    with st.active(trace):
        st.record("model", "output[g]", {"out": x})
    x.add_(1.0)
    assert trace.stages[0].tensors["out"].sum() == 0


def test_repeated_stage_keeps_the_first_calls_and_the_last() -> None:
    trace = st.SnapshotTrace(phase="val", step=0, max_repeats=2)
    with st.active(trace):
        for i in range(5):
            st.record("model", "input[g]", {"x": torch.full((1, 1, 2, 2), float(i))})
    names = [s.name for s in trace.stages]
    assert names == ["input[g]", "input[g]#1", "input[g]#last"]
    assert trace.stages[-1].tensors["x"].max() == 4


def test_unbatched_trace_keeps_the_whole_tensor() -> None:
    trace = st.SnapshotTrace(phase="train", step=0, batched=False)
    with st.active(trace):
        st.record("loader", "read", {"kspace": torch.ones(4, 8, 8)})
    assert trace.stages[0].tensors["kspace"].shape == (4, 8, 8)


def test_suspended_hides_the_trace() -> None:
    trace = st.SnapshotTrace(phase="train", step=0)
    with st.active(trace), st.suspended():
        st.record("batch", "x", {"x": torch.ones(1)})
    assert trace.stages == []
