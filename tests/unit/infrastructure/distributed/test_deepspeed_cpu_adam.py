"""The offload path's CPU Adam: the torch-to-DeepSpeedCPUAdam mapping, refusals and provenance.

DeepSpeedCPUAdam is replaced by a recording fake (the C++ op is not built on CI), so
these tests pin the mapping; the real construction is owed on the GPU lane (b4 arm).
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
from torch import nn

import spectramr.infrastructure.distributed.deepspeed_backend.cpu_adam as cpu_adam
import spectramr.infrastructure.distributed.deepspeed_backend.runtime as runtime
from spectramr.infrastructure.distributed.deepspeed_backend.cpu_adam import (
    prepare_client_optimizer,
)

pytestmark = pytest.mark.unit


class _RecordingCPUAdam:
    """Stands in for DeepSpeedCPUAdam: records what it was constructed with."""

    instances: ClassVar[list[_RecordingCPUAdam]] = []

    def __init__(self, groups, **kwargs) -> None:
        self.groups = groups
        self.kwargs = kwargs
        self.param_groups = groups
        _RecordingCPUAdam.instances.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.instances = []


@pytest.fixture
def fake_cls(monkeypatch):
    _RecordingCPUAdam.reset()
    monkeypatch.setattr(cpu_adam, "cpu_adam_class", lambda *, zenflow: _RecordingCPUAdam)
    monkeypatch.setattr(runtime, "require_deepspeed", lambda: object())
    return _RecordingCPUAdam


def _offload(optimizer, scheduler=None, *, zenflow=False, config=None):
    return prepare_client_optimizer(
        optimizer,
        scheduler,
        config=config or SimpleNamespace(optimization=None, training=None),
        zero_stage=2,
        offload_optimizer="cpu",
        zenflow=zenflow,
    )


class TestMapping:
    def test_two_ttur_groups_keep_their_own_hyperparameters(self, fake_cls) -> None:
        a, b = nn.Linear(2, 2), nn.Linear(2, 2)
        opt = torch.optim.AdamW(
            [
                {"params": a.parameters(), "lr": 1e-4, "weight_decay": 0.01},
                {"params": b.parameters(), "lr": 4e-4, "weight_decay": 0.0},
            ],
            betas=(0.5, 0.999),
            eps=1e-7,
        )
        client, _, record = _offload(opt)
        assert isinstance(client, _RecordingCPUAdam)
        groups = client.groups
        assert [g["lr"] for g in groups] == [1e-4, 4e-4]
        assert [g["weight_decay"] for g in groups] == [0.01, 0.0]
        assert all(g["betas"] == (0.5, 0.999) and g["eps"] == 1e-7 for g in groups)
        assert [list(g["params"]) for g in groups] == [list(a.parameters()), list(b.parameters())]
        assert client.kwargs["betas"] == (0.5, 0.999) and client.kwargs["eps"] == 1e-7
        assert client.kwargs["amsgrad"] is False
        assert record["client"] == "_RecordingCPUAdam"
        assert record["source"] == "torch.optim.adamw.AdamW"
        assert record["param_groups"] == 2

    def test_adam_maps_to_l2_mode_and_adamw_to_decoupled(self, fake_cls) -> None:
        params = nn.Linear(2, 2).parameters()
        client, _, record = _offload(torch.optim.Adam(params, lr=1e-3))
        assert client.kwargs["adamw_mode"] is False
        assert record["adamw_mode"] is False
        client, _, record = _offload(torch.optim.AdamW(nn.Linear(2, 2).parameters(), lr=1e-3))
        assert client.kwargs["adamw_mode"] is True
        assert record["adamw_mode"] is True

    def test_zenflow_selects_the_zenflow_class(self, monkeypatch) -> None:
        seen = {}

        def pick(*, zenflow):
            seen["zenflow"] = zenflow
            return _RecordingCPUAdam

        monkeypatch.setattr(cpu_adam, "cpu_adam_class", pick)
        _offload(torch.optim.AdamW(nn.Linear(2, 2).parameters()), zenflow=True)
        assert seen == {"zenflow": True}


class TestRefusals:
    def test_planted_sgd_under_offload_is_refused_before_construction(self, fake_cls) -> None:
        opt = torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=1e-2)
        with pytest.raises(ValueError, match="rebuilt as DeepSpeedCPUAdam"):
            _offload(opt)
        assert fake_cls.instances == []

    @pytest.mark.parametrize("knob", ["amsgrad", "maximize"])
    def test_planted_true_update_rule_knob_is_refused(self, fake_cls, knob) -> None:
        opt = torch.optim.Adam(nn.Linear(2, 2).parameters())
        opt.param_groups[0][knob] = True
        with pytest.raises(ValueError, match=knob):
            _offload(opt)
        assert fake_cls.instances == []

    def test_unmapped_group_key_is_refused_not_dropped(self, fake_cls) -> None:
        opt = torch.optim.Adam(nn.Linear(2, 2).parameters())
        opt.param_groups[0]["momentum_warp"] = 3
        with pytest.raises(ValueError, match="momentum_warp"):
            _offload(opt)

    def test_a_subclass_of_adam_is_refused(self, fake_cls) -> None:
        class MyAdam(torch.optim.Adam):
            pass

        with pytest.raises(ValueError, match="rebuilt as DeepSpeedCPUAdam"):
            _offload(MyAdam(nn.Linear(2, 2).parameters()))


class TestPassThroughAndScheduler:
    def test_no_offload_returns_the_torch_optimizer_untouched(self, fake_cls) -> None:
        opt = torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=1e-2)
        client, sched, record = prepare_client_optimizer(
            opt,
            None,
            config=None,
            zero_stage=0,
            offload_optimizer="none",
            zenflow=False,
        )
        assert client is opt and sched is None
        assert fake_cls.instances == []
        assert record["client"] == "torch.optim.sgd.SGD"

    def test_scheduler_is_rebuilt_on_the_cpu_optimizer(self, fake_cls, monkeypatch) -> None:
        import spectramr.infrastructure.training.scheduler_resolution as sr

        spec = object()
        monkeypatch.setattr(cpu_adam, "resolve_scheduler_spec", lambda opt, **kw: spec)
        monkeypatch.setattr(
            cpu_adam,
            "build_scheduler_from_spec",
            lambda s, optimizer: ("rebuilt", s, optimizer),
        )
        assert sr is not None
        opt = torch.optim.Adam(nn.Linear(2, 2).parameters())
        config = SimpleNamespace(optimization=object(), training=SimpleNamespace(max_iterations=10))
        client, sched, _ = _offload(opt, scheduler="old", config=config)
        assert sched == ("rebuilt", spec, client)

    def test_a_missing_scheduler_stays_missing(self, fake_cls) -> None:
        _, sched, _ = _offload(torch.optim.Adam(nn.Linear(2, 2).parameters()))
        assert sched is None


class TestConstructionFailures:
    def test_jit_build_failure_raises_with_the_hint(self, monkeypatch) -> None:
        class Broken:
            def __init__(self, *a, **k):
                raise OSError("ninja: command not found")

        monkeypatch.setattr(cpu_adam, "cpu_adam_class", lambda *, zenflow: Broken)
        with pytest.raises(RuntimeError, match="DS_BUILD_CPU_ADAM") as info:
            _offload(torch.optim.Adam(nn.Linear(2, 2).parameters()))
        assert "ninja" in str(info.value)

    def test_missing_op_module_raises_with_the_hint_not_a_torch_fallback(self, monkeypatch) -> None:
        monkeypatch.setattr(runtime, "require_deepspeed", lambda: object())
        monkeypatch.setitem(sys.modules, "deepspeed.ops.adam", None)
        with pytest.raises(ImportError, match="DS_BUILD_CPU_ADAM"):
            runtime.cpu_adam_class(zenflow=False)


class TestAdoptWiring:
    """adopt() hands DeepSpeed the CPU optimizer and stamps the class that ran."""

    def test_offload_adopt_passes_the_cpu_optimizer_and_records_it(
        self, fake_cls, monkeypatch
    ) -> None:
        import spectramr.infrastructure.distributed.deepspeed_backend as backend
        import spectramr.infrastructure.distributed.strategies as strategies
        from spectramr.config.schemas.base import ParallelismConfigSchema

        handed = {}

        class _Engine:
            pass

        def fake_initialize(*, model, optimizer, lr_scheduler, config):
            handed["optimizer"] = optimizer
            ds_wrapper = type("ZeROWrapper", (), {})()
            return _Engine(), ds_wrapper, None

        monkeypatch.setattr(strategies, "_require_process_group", lambda _name: None)
        monkeypatch.setattr(backend, "initialize_deepspeed_engine", fake_initialize)
        monkeypatch.setattr(backend, "build_deepspeed_config", lambda settings: {})
        parallel = ParallelismConfigSchema(
            strategy="deepspeed",
            deepspeed={"enabled": True, "zero_stage": 2, "offload_optimizer": "cpu"},
        )
        ctx = SimpleNamespace(
            config=SimpleNamespace(optimization=None, training=None),
            parallel=parallel,
            world_size=1,
            device=torch.device("cpu"),
        )
        torch_opt = torch.optim.AdamW(nn.Linear(2, 2).parameters(), lr=1e-3)
        result = strategies.DeepSpeedStrategy().adopt(
            {"generator": nn.Linear(2, 2)}, {"opt_g": torch_opt}, {}, ctx
        )
        assert isinstance(handed["optimizer"], _RecordingCPUAdam)
        record = result.provenance["optimizers"]["opt_g"]
        assert record["client"] == "_RecordingCPUAdam"
        assert record["source"] == "torch.optim.adamw.AdamW"
        assert record["engine"] == "ZeROWrapper"
