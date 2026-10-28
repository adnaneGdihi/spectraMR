"""Diagnostic and validation probes that read the wrapped generator under DDP.

Each scenario runs the production method under a real two-rank gloo DDP. Red before the fix,
green after: the probe sees the inner module's attribute."""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.unit

WORLD = 2


class _Rec:
    """Logging stub that keeps every warning, so a test can assert on what was emitted."""

    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.logger = SimpleNamespace(isEnabledFor=lambda level: True)

    def log_info(self, msg, *args, **kwargs):
        pass

    def log_debug(self, msg, *args, **kwargs):
        pass

    def log_warning(self, msg, *args, **kwargs):
        self.warnings.append(str(msg))


def _ddp(module):
    from torch.nn.parallel import DistributedDataParallel

    return DistributedDataParallel(module)


class _Linear(torch.nn.Module):
    """Parameter-bearing base: DDP refuses a module without trainable parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x, *args, **kwargs):
        return self.conv(x)


class _ScoreFree(_Linear):
    """Accepts no timesteps: a strict forward, so an injected kwarg is a TypeError."""

    def forward(self, x_t):
        return self.conv(x_t)


class _TimeNet(_Linear):
    def __init__(self) -> None:
        super().__init__()
        self.time_embedding = torch.nn.Linear(1, 1)
        self.synthetic_forward_probe_skip = set()


class _SimNet(_Linear):
    def __init__(self) -> None:
        super().__init__()
        self.simulator = SimpleNamespace(progressive_degradations=_Degrads([1, 2]))


class _Degrads(list):
    copied = False

    def copy(self):
        type(self).copied = True
        return list(self)


class _StopProbeError(Exception):
    """Ends a production method once the probe under test has been exercised."""


def _check_debug_time_attr_seen_through_ddp(rank: int) -> None:
    """The identity-collapse diagnostic reads the time attribute and the probe skip set inner."""
    from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy

    net = _TimeNet()
    net.synthetic_forward_probe_skip = {"identity_collapse"}
    rec = _Rec()
    s = object.__new__(DiffusionTrainingStrategy)
    s.logging_service = rec
    s.loop_state = SimpleNamespace(iteration=0)
    s.env = SimpleNamespace(generator=_ddp(net), losses={})
    s.config = SimpleNamespace(
        undersampling=None,
        logging=SimpleNamespace(
            snapshots=SimpleNamespace(log_steps=[]),
            intervals=SimpleNamespace(anomaly_check=1),
        ),
        training=SimpleNamespace(
            identity_collapse_threshold=0.01,
            early_training_steps=0,
        ),
    )
    x = torch.rand(1, 1, 4, 4)
    s._compute_and_log_debug_info(x, x, x, x)
    assert not any("No known time-conditioning" in w for w in rec.warnings), rec.warnings
    assert not any("IDENTITY COLLAPSE" in w for w in rec.warnings), rec.warnings


def _check_simulator_probe_seen_through_ddp(rank: int) -> None:
    """The progressive-degradation branch runs when the wrapped generator exposes a simulator."""
    from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy

    x = torch.rand(1, 1, 4, 4)
    s = object.__new__(DiffusionTrainingStrategy)
    s.num_timesteps = 10
    s.config = SimpleNamespace(
        undersampling=SimpleNamespace(base_acceleration=1.0, max_acceleration=8.0),
        model=SimpleNamespace(model_type="image_cold_diffusion"),
    )
    s.env = SimpleNamespace(generator=_ddp(_SimNet()), losses={})
    s._prepare_validation_data = lambda *a: (x, x, torch.ones(1))

    def _stop(*args, **kwargs):
        raise _StopProbeError

    s._generate_validation_prediction = _stop
    _Degrads.copied = False
    with pytest.raises(_StopProbeError):
        s.get_validation_images(batch={}, input_batch=x, target_batch=x)
    assert _Degrads.copied, "progressive-degradation branch was skipped under DDP"


def _check_x_diffusion_forward_kwargs_gated_by_inner(rank: int) -> None:
    from spectramr.infrastructure.training.strategies.diffusion import (
        XDiffusionTrainingStrategy,
    )

    s = object.__new__(XDiffusionTrainingStrategy)
    s.logging_service = _Rec()
    s.config = SimpleNamespace(
        model=SimpleNamespace(model_type="x_diffusion"),
        training=SimpleNamespace(diffusion=SimpleNamespace(timesteps=10)),
    )
    s.env = SimpleNamespace(generator=_ddp(_ScoreFree()), losses={})
    s._declare_model_input = lambda *a, **k: None
    s._loss_dict_reuse = {}
    x = torch.rand(1, 1, 4, 4)
    out = s._compute_losses_impl(x, x, epoch=0)
    assert out is not None


SCENARIOS = {
    "debug_probes": _check_debug_time_attr_seen_through_ddp,
    "simulator": _check_simulator_probe_seen_through_ddp,
    "x_diffusion_gen_forward": _check_x_diffusion_forward_kwargs_gated_by_inner,
}


def _worker(rank: int, store: str, scenario: str) -> None:
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=WORLD)
    try:
        SCENARIOS[scenario](rank)
    finally:
        dist.destroy_process_group()


def _spawn(scenario: str) -> None:
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_worker, args=(os.path.join(tmp, "store"), scenario), nprocs=WORLD, join=True)


def test_debug_probes_read_the_inner_module_under_ddp() -> None:
    _spawn("debug_probes")


def test_simulator_probe_reads_the_inner_module_under_ddp() -> None:
    _spawn("simulator")


def test_x_diffusion_gen_forward_is_gated_by_inner_under_ddp() -> None:
    _spawn("x_diffusion_gen_forward")
