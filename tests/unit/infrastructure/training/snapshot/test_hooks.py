"""Forward hooks record what a model received and returned, once per call, only when traced."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from spectramr.core import snapshot_trace as st
from spectramr.core.snapshot_trace import View
from spectramr.infrastructure.training.snapshot.hooks import ModelHooks, flatten_tensors

pytestmark = pytest.mark.unit


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 1)

    def forward(
        self, x: torch.Tensor, *, t: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict]:
        y = self.conv(x)
        return y, {"aux": y * 2}


class _Wrapper(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.module = inner

    def forward(self, *a, **k):
        return self.module(*a, **k)


def _hooks(models: dict) -> ModelHooks:
    return ModelHooks(models, lambda kind: View())


def test_nothing_is_recorded_outside_a_trace() -> None:
    net = _Net()
    _hooks({"generator": net})
    net(torch.ones(1, 2, 4, 4))  # must not raise or record


def test_input_and_every_output_tensor_are_recorded() -> None:
    net = _Net()
    _hooks({"generator": net})
    trace = st.SnapshotTrace(phase="train", step=0)
    with st.active(trace):
        net(torch.ones(1, 2, 4, 4), t=torch.tensor([3.0]))
    inp, out = trace.stages
    assert (inp.group, inp.name) == ("model", "input[generator]")
    assert set(inp.tensors) == {"arg.0", "kw.t"}
    assert set(out.tensors) == {"0", "1.aux"}


def test_a_wrapper_and_its_inner_module_record_one_call() -> None:
    inner = _Net()
    wrapped = _Wrapper(inner)
    _hooks({"generator": wrapped})
    trace = st.SnapshotTrace(phase="train", step=0)
    with st.active(trace):
        wrapped(torch.ones(1, 2, 4, 4))
        inner(torch.ones(1, 2, 4, 4))  # a strategy that calls the unwrapped module
    assert [s.name for s in trace.stages] == [
        "input[generator]",
        "output[generator]",
        "input[generator]#1",
        "output[generator]#1",
    ]


def test_a_raising_forward_does_not_leave_the_hook_suppressed() -> None:
    net = _Net()
    _hooks({"generator": net})
    trace = st.SnapshotTrace(phase="train", step=0)
    with st.active(trace):
        with pytest.raises(RuntimeError):
            net(torch.ones(1, 3, 4, 4))  # wrong channel count
        net(torch.ones(1, 2, 4, 4))
    assert "output[generator]" in [s.name for s in trace.stages]


def test_declared_segments_apply_only_to_the_tensor_they_describe() -> None:
    net = _Net()
    _hooks({"generator": net})
    trace = st.SnapshotTrace(phase="train", step=0)
    trace.model_views["input"] = View(kspace=True, segments=(("a", 1, True), ("b", 1, False)))
    with st.active(trace):
        net(torch.ones(1, 2, 4, 4), t=torch.tensor([1.0]))
    views = trace.stages[0].views
    assert views["arg.0"].segments is not None
    assert views["kw.t"].segments is None


def test_flatten_names_nested_structures() -> None:
    got = flatten_tensors({"a": [torch.ones(1), {"b": torch.zeros(1)}]}, "kw")
    assert set(got) == {"kw.a.0", "kw.a.1.b"}


def test_a_module_attached_later_is_recorded_once() -> None:
    """A strategy-owned generator registers itself; a second attach is a no-op."""
    net = _Net()
    hooks = _hooks({})
    hooks.attach("CycleGenerator", net)
    hooks.attach("CycleGenerator", net)
    trace = st.SnapshotTrace(phase="train", step=0)
    with st.active(trace):
        net(torch.ones(1, 2, 4, 4))
    assert [s.name for s in trace.stages] == ["input[CycleGenerator]", "output[CycleGenerator]"]


def test_hooks_do_nothing_while_dynamo_traces(monkeypatch) -> None:
    net = _Net()
    _hooks({"generator": net})
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    trace = st.SnapshotTrace(phase="train", step=0)
    with st.active(trace):
        net(torch.ones(1, 2, 4, 4))
    assert trace.stages == []


class _DC(nn.Module):
    snapshot_stage = "post"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 0.5


class _Unrolled(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Conv2d(1, 1, 1)
        self.dc = _DC()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = self.dc(self.body(x))
        return x


def test_a_post_stage_module_inside_a_model_is_recorded_per_call() -> None:
    net = _Unrolled()
    _hooks({"generator": net})
    trace = st.SnapshotTrace(phase="train", step=0, max_repeats=2)
    with st.active(trace):
        net(torch.ones(1, 1, 4, 4))
    names = [(s.group, s.name) for s in trace.stages]
    assert ("post", "input[_DC]") in names and ("post", "output[_DC]#1") in names
    assert ("post", "output[_DC]#last") in names, "the third cascade overwrites the last slot"
    assert names[0] == ("model", "input[generator]") and names[-1] == ("model", "output[generator]")


def test_physics_modules_outside_the_model_are_hooked_too() -> None:
    dc = _DC()
    hooks = _hooks({})
    hooks.attach_stages(dc)
    trace = st.SnapshotTrace(phase="train", step=0)
    with st.active(trace):
        dc(torch.ones(1, 1, 2, 2))
    assert [s.name for s in trace.stages] == ["input[_DC]", "output[_DC]"]
    assert all(s.group == "post" for s in trace.stages)


def test_every_tagged_class_is_a_module_that_declares_post() -> None:
    """Planted-violation guard: a typo'd stage name would record under a stray group."""
    import importlib

    tagged = {
        "spectramr.infrastructure.physics.data_consistency": [
            "HardDataConsistency",
            "SoftDataConsistency",
            "NoiseAdaptiveDataConsistency",
        ],
        "spectramr.infrastructure.physics.integration": ["OperatorProjectionDataConsistency"],
        "spectramr.infrastructure.physics.forward_operator": ["NullSpaceProjection"],
    }
    for mod, classes in tagged.items():
        m = importlib.import_module(mod)
        for c in classes:
            assert getattr(m, c).snapshot_stage == "post", c
