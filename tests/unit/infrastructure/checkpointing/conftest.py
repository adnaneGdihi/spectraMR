"""A synthetic RunState carrying every part a training checkpoint holds."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch import nn

from spectramr.infrastructure.checkpointing.payload import build_meta


class Tied(nn.Module):
    """An embedding tied to its output head, the shape safetensors refuses unshared."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(5, 3)
        self.head = nn.Linear(3, 5, bias=False)
        self.head.weight = self.embed.weight


def make_run_state(fmt: str, *, role: str = "step") -> dict:
    torch.manual_seed(0)
    gen = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    opt = torch.optim.Adam(gen.parameters(), lr=1e-3)
    gen(torch.randn(2, 4)).sum().backward()
    opt.step()
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10)
    best = {"name": "psnr", "value": 30.25} if role == "best" else None
    return {
        "meta": build_meta(
            role=role,
            reason="best" if role == "best" else "periodic",
            epoch=2,
            step=150,
            fmt=fmt,
            topology={"world_size": 1, "parallel_strategy": "none"},
            config_fingerprint="cafe",
            best_metric=best,
        ),
        "models": {"generator": gen.state_dict(), "stage:tied": Tied().state_dict()},
        "ema": {"generator": {k: v.clone() for k, v in gen.state_dict().items()}, "num_updates": 7},
        "optim": {"generator": opt.state_dict()},
        "sched": {"generator": sched.state_dict()},
        "rng": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
        "strategy": {"params": {"sigma": torch.tensor(0.5)}, "modules": {}},
        "early_stopping": {
            "monitor": "val/psnr",
            "mode": "max",
            "best_value": float("nan"),
            "best_iteration": 100,
            "wait_count": 0,
            "stop_signal": False,
        },
    }


@pytest.fixture
def run_state():
    return make_run_state


class FakeEarlyStopping:
    def __init__(self, state: dict | None = None) -> None:
        self.state = state or {
            "monitor": "val_psnr",
            "mode": "max",
            "best_value": 21.5,
            "best_iteration": 4,
            "wait_count": 1,
            "stop_signal": False,
        }

    def state_dict(self) -> dict:
        return dict(self.state)


def make_pipeline(*, discriminator: bool = True, ema: bool = True, scaler: bool = False):
    from types import SimpleNamespace

    from spectramr.infrastructure.optimization.ema import ModelEma

    torch.manual_seed(0)
    gen = nn.Linear(4, 2)
    disc = nn.Linear(2, 1) if discriminator else None
    opt_g = torch.optim.Adam(gen.parameters(), lr=1e-3)
    opt_d = torch.optim.Adam(disc.parameters(), lr=1e-3) if disc is not None else None
    gen(torch.randn(3, 4)).sum().backward()
    opt_g.step()
    if disc is not None:
        disc(torch.randn(3, 2)).sum().backward()
        opt_d.step()
    shadow = ModelEma(gen, decay=0.9, device=torch.device("cpu")) if ema else None
    if shadow is not None:
        shadow.update(gen)
    return SimpleNamespace(
        generator=gen,
        discriminator=disc,
        optimizer_g=opt_g,
        optimizer_d=opt_d,
        scheduler_g=torch.optim.lr_scheduler.StepLR(opt_g, step_size=2),
        scheduler_d=None,
        ema=shadow,
        scaler=torch.amp.GradScaler("cpu", enabled=True) if scaler else None,
        device="cpu",
    )


@pytest.fixture
def tiny_pipeline():
    return make_pipeline


class FakeNativeAdapter:
    """A native-artifact adapter implementing the whole checkpoint-adapter Protocol.

    ``load_native`` records its keywords in ``seen`` and returns ``handled``; a callable
    ``on_load`` replaces that, so a test can plant a failure inside the native load.
    """

    name = "deepspeed"
    requires_all_ranks = True
    writes_native_artifact = True

    def __init__(
        self,
        handled: frozenset[str] = frozenset({"generator"}),
        *,
        ema_whole: bool = True,
        on_load=None,
    ) -> None:
        from spectramr.infrastructure.distributed.state_io import PassthroughStateIO

        self.state_io = PassthroughStateIO()
        self.handled = handled
        self.ema_whole = ema_whole
        self.on_load = on_load
        self.seen: dict = {}

    def full_model_state(self, model):
        return self.state_io.model_state(model)

    def engine_owned_keys(self, models) -> frozenset[str]:
        return self.handled & frozenset(models)

    def ema_is_whole(self, models) -> bool:
        return self.ema_whole

    def save_native(self, **_kwargs):
        return None

    def load_native(self, **kwargs):
        from spectramr.infrastructure.distributed.checkpoint_adapters import NativeLoadResult

        self.seen.update(kwargs)
        if self.on_load is not None:
            return self.on_load(**kwargs)
        return NativeLoadResult({"epoch": 1, "global_step": 4}, self.handled)

    def native_tag_roots(self, **_kwargs):
        return ()


def native_runtime(adapter: FakeNativeAdapter, strategy: str = "deepspeed"):
    from types import SimpleNamespace

    return SimpleNamespace(strategy=strategy, checkpoint_adapter=adapter)


def capture_with(pipeline, **kw):
    """``capture`` with a step file's defaults; keywords override them."""
    from spectramr.infrastructure.checkpointing.capture import capture

    args = {
        "strategy": None,
        "parallel": None,
        "early_stopping": None,
        "role": "step",
        "reason": "periodic",
        "epoch": 1,
        "step": 4,
        "fmt": "pt",
    }
    args.update(kw)
    return capture(pipeline, **args)
