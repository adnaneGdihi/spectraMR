"""Two-stage MultiTrainingStrategy fakes shared by the stage-checkpoint tests (#2962).

The strategy is built with ``object.__new__`` and two ``nn.Linear`` stages: a real
``__init__`` builds every stage through ``GeneratorBuilder``. ``env.opt_g`` is the
pipeline's ``optimizer_g`` object, as in a real run, so ``resume`` loads the optimizer
the stages registered their parameters on.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from spectramr.infrastructure.checkpointing import store
from spectramr.infrastructure.checkpointing.layout import CheckpointLayout
from spectramr.infrastructure.checkpointing.resume import resume
from spectramr.infrastructure.training.strategies.pipeline_strategy import MultiTrainingStrategy
from tests.unit.infrastructure.checkpointing.conftest import make_pipeline

FORMATS = ("safetensors", "pt", "pth")


def _strategy(
    pipeline, *, seed: int, frozen_at_init: tuple[str, ...] = ()
) -> MultiTrainingStrategy:
    """What ``_setup_strategy_specific_components`` leaves behind for two stages.

    Stage ``b`` has its own optimizer; stage ``a`` is driven by ``opt_g`` and carries
    an early-stopping block.
    """
    torch.manual_seed(seed)
    s = object.__new__(MultiTrainingStrategy)
    s.env = SimpleNamespace(opt_g=pipeline.optimizer_g, generator=pipeline.generator)
    s.device = torch.device("cpu")
    s.multi_stages = {"a": nn.Linear(4, 4), "b": nn.Linear(4, 4)}
    for name in frozen_at_init:
        s.multi_stages[name].requires_grad_(False)
    s._stage_optimizers = {"b": torch.optim.Adam(s.multi_stages["b"].parameters(), lr=1e-2)}
    s._finetune_epoch = 2
    s._stage_early_stopping = {"a": SimpleNamespace(enabled=True, patience=1, metric="val_a_l1")}
    s._es_unresolved_monitors = set()
    s._trainable_params = s._collect_trainable_params()
    s._register_global_params()
    return s


def _train_step(pipeline, s: MultiTrainingStrategy) -> None:
    x = torch.randn(3, 4)
    loss = pipeline.generator(x).sum() + s.multi_stages["b"](s.multi_stages["a"](x)).sum()
    loss.backward()
    pipeline.optimizer_g.step()
    s._stage_optimizers["b"].step()


def _config(fmt: str, **gates: bool) -> SimpleNamespace:
    return SimpleNamespace(checkpoint=SimpleNamespace(format=fmt, keep_last_n=5, **gates))


def _save(tmp_path: Path, pipeline, s, *, fmt: str = "pt") -> Path:
    return store.save_step(
        pipeline,
        layout=CheckpointLayout.for_run(tmp_path),
        config=_config(fmt),
        strategy=s,
        parallel=None,
        early_stopping=None,
        reason="periodic",
        epoch=3,
        step=12,
    )


def _resume(path: Path, pipeline, s, *, fmt: str = "pt", **gates: bool):
    return resume(path, pipeline, strategy=s, parallel=None, config=_config(fmt, **gates))


def _trained(*, frozen_at_init: tuple[str, ...] = ()):
    pipeline = make_pipeline(discriminator=False, ema=False)
    s = _strategy(pipeline, seed=1, frozen_at_init=frozen_at_init)
    _train_step(pipeline, s)
    return pipeline, s


def _equal(a: nn.Module, b: nn.Module) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
