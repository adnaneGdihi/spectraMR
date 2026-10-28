"""capture() under a parallel runtime: the best file's EMA, ZeRO gathers, engine state."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from spectramr.infrastructure.checkpointing import codecs
from spectramr.infrastructure.distributed.checkpoint_adapters import (
    DeepSpeedCheckpointAdapter,
    DefaultCheckpointAdapter,
)
from tests.unit.infrastructure.checkpointing.conftest import make_pipeline
from tests.unit.infrastructure.checkpointing.test_capture import _capture


class TestBestEma:
    def _parallel(self, whole: bool):
        adapter = DefaultCheckpointAdapter()
        adapter.ema_is_whole = lambda models: whole
        return SimpleNamespace(strategy="deepspeed", checkpoint_adapter=adapter)

    def _best(self, parallel, pipeline=None):
        return _capture(
            pipeline or make_pipeline(),
            parallel=parallel,
            role="best",
            reason="best",
            best_metric={"name": "val_psnr", "value": 1.0},
        )

    def test_whole_parameter_shadow_is_in_the_best_file(self) -> None:
        assert "ema" in self._best(self._parallel(True))

    def test_partitioned_shadow_is_not(self) -> None:
        assert "ema" not in self._best(self._parallel(False))

    def test_a_step_file_carries_the_shadow_regardless(self) -> None:
        assert "ema" in _capture(make_pipeline(), parallel=self._parallel(False))

    def test_no_ema_means_no_part(self) -> None:
        assert "ema" not in self._best(self._parallel(True), make_pipeline(ema=False))


def test_zero3_capture_gathers_through_the_adapter_on_every_rank() -> None:
    """Non-writer ranks get None from the gather and still produce a valid RunState."""
    gathered: list[str] = []

    class Zero3(DefaultCheckpointAdapter):
        def full_model_state(self, model):
            gathered.append(type(model).__name__)
            return {}  # what a non-zero rank receives

    state = _capture(
        make_pipeline(),
        parallel=SimpleNamespace(strategy="deepspeed", checkpoint_adapter=Zero3()),
    )
    assert gathered == ["Linear", "Linear"]
    assert state["models"]["generator"] == {}


class _ZeroOptimizer:
    """A ZeRO optimizer's state dict holds DeepSpeed's LossScaler, which no codec stores."""

    def state_dict(self) -> dict:
        return {"loss_scaler": object(), "base_optimizer_state": {}}


class _Engine:
    """Duck-types a DeepSpeed engine below ZeRO-3: ``save_checkpoint`` and ``.module``."""

    def __init__(self) -> None:
        self.module = nn.Linear(4, 2)

    def save_checkpoint(self, *a, **k):
        raise AssertionError("not called by capture")


def test_an_engine_owned_optimizer_and_scheduler_stay_in_the_native_tag(tmp_path: Path) -> None:
    pipe = make_pipeline()
    pipe.generator = _Engine()
    pipe.optimizer_g = _ZeroOptimizer()
    parallel = SimpleNamespace(
        strategy="deepspeed", checkpoint_adapter=DeepSpeedCheckpointAdapter()
    )
    state = _capture(pipe, parallel=parallel)
    assert "generator" not in state["optim"]
    assert "discriminator" in state["optim"]
    assert "generator" not in state.get("sched", {})
    assert set(state["models"]["generator"]) == {"weight", "bias"}
    codecs.write(tmp_path / "step_00000004.pt", state)


def test_a_disabled_scaler_writes_no_part() -> None:
    """A disabled GradScaler's state is ``{}``, which ``load_state_dict`` refuses on resume."""
    pipe = make_pipeline()
    pipe.scaler = torch.amp.GradScaler("cpu", enabled=False)
    assert pipe.scaler.state_dict() == {}
    assert "scaler" not in _capture(pipe)
